"""Train and compare the three CNN models for DIT5411.

Usage examples:
    # Full assignment run (needs the 13,065-character dataset)
    python train.py --data-dir data/cleaned_data --epochs 15

    # Quick pilot run on 20 characters, 2 epochs
    python train.py --num-classes 20 --epochs 2 --models A B

    # Offline test (no ImageNet download for Model C)
    python train.py --num-classes 20 --epochs 2 --no-imagenet
"""
import argparse
import json
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from dataset import split_dataset, save_label_map
from generators import HandwritingSequence
from models import get_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data/cleaned_data')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--num-classes', type=int, default=None,
                   help='Limit number of characters (pilot run). Default: all 13,065')
    p.add_argument('--num-train', type=int, default=40,
                   help='First N samples per character used for training')
    p.add_argument('--augment-factor', type=int, default=5,
                   help='Versions per original (40x5=200 per character)')
    p.add_argument('--models', nargs='+', default=['A', 'B', 'C'],
                   help='Which models to train: A B C')
    p.add_argument('--no-imagenet', action='store_true',
                   help='Model C uses random weights instead of downloading ImageNet')
    p.add_argument('--steps-per-epoch', type=int, default=None,
                   help='Cap steps per epoch (e.g. 3000). Prevents Colab free-tier '
                        'disconnect mid-epoch; still keeps 200 augmented samples/class.')
    p.add_argument('--resume', action='store_true',
                   help='Resume each model from its latest saved checkpoint.')
    p.add_argument('--mixed-precision', action='store_true',
                   help='Use mixed_float16 (much faster on T4 GPU).')
    p.add_argument('--validation-steps', type=int, default=None,
                   help='Cap validation steps (faster per epoch).')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


class TqdmProgress(keras.callbacks.Callback):
    """Live tqdm progress bar that renders in Kaggle/Colab notebook output."""

    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps
        self.bar = None
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.bar = tqdm(total=self.total_steps, desc=f"Epoch {self.epoch}",
                        unit="step", ncols=80, position=0, leave=True)

    def on_train_batch_end(self, batch, logs=None):
        if self.bar is not None:
            logs = logs or {}
            self.bar.set_postfix(loss=f"{logs.get('loss', 0):.4f}",
                                 acc=f"{logs.get('accuracy', 0)*100:.1f}%")
            self.bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class ProgressPrinter(keras.callbacks.Callback):
    """Print a plain progress line every N steps so it is always visible
    in Kaggle/Colab output (Keras' animated bar often does not render)."""

    def __init__(self, every=100):
        super().__init__()
        self.every = every

    def on_epoch_begin(self, epoch, logs=None):
        print(f"    ===== EPOCH {epoch + 1} STARTED =====", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        logs = logs or {}
        if step == 1 or step % self.every == 0:
            print(f"    step {step} - loss {logs.get('loss', 0):.4f} - "
                  f"acc {logs.get('accuracy', 0)*100:.2f}%", flush=True)


class EpochCheckpoint(keras.callbacks.Callback):
    """Save model + completed-epoch count after every epoch so a Colab
    disconnect never loses more than the current (partial) epoch."""

    def __init__(self, model_path, state_path, history):
        super().__init__()
        self.model_path = model_path
        self.state_path = state_path
        self.history = history

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_path)
        self.history['accuracy'].append(round(float(logs['accuracy']), 5))
        self.history['val_accuracy'].append(round(float(logs['val_accuracy']), 5))
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump({
                'completed_epochs': epoch + 1,
                'history': self.history,
            }, f, indent=2)
        print(f"  [checkpoint saved: epoch {epoch + 1}]", flush=True)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print('      Mixed precision: mixed_float16', flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print('[1/4] Splitting dataset...', flush=True)
    splits = split_dataset(args.data_dir, args.num_train, args.num_classes)
    chars = sorted(splits.keys(), key=lambda x: x.encode('utf-8'))
    if not chars:
        raise SystemExit('No character data found. Check --data-dir.')

    char2idx = {c: i for i, c in enumerate(chars)}
    num_classes = len(chars)
    label_map_path = os.path.join(args.output_dir, 'label_map.json')
    save_label_map(chars, label_map_path)
    print(f'      Characters: {num_classes}', flush=True)

    train_records, test_records = [], []
    for ch in chars:
        idx = char2idx[ch]
        train_records.extend((p, idx) for p in splits[ch]['train'])
        test_records.extend((p, idx) for p in splits[ch]['test'])
    print(f'      Original train images: {len(train_records)}', flush=True)
    print(f'      Test images: {len(test_records)}', flush=True)
    print(f'      Augmented train size: {len(train_records) * args.augment_factor}',
          flush=True)

    print('[2/4] Building generators...', flush=True)
    train_gen = HandwritingSequence(
        train_records, args.batch_size, num_classes, args.img_size,
        factor=args.augment_factor, shuffle=True, seed=args.seed)
    test_gen = HandwritingSequence(
        test_records, args.batch_size, num_classes, args.img_size,
        factor=1, shuffle=False, seed=args.seed)

    results = {
        'assignment': 'DIT5411 Character Recognition',
        'num_classes': num_classes,
        'img_size': args.img_size,
        'num_train_per_char': args.num_train,
        'augment_factor': args.augment_factor,
        'train_samples_augmented': len(train_records) * args.augment_factor,
        'test_samples': len(test_records),
        'epochs': args.epochs,
        'models': [],
    }

    # Merge with previous results so training models in separate cells/runs
    # accumulates all models into one comparison table instead of overwriting.
    results_path = os.path.join(args.output_dir, 'results.json')
    if os.path.isfile(results_path):
        try:
            with open(results_path, encoding='utf-8') as f:
                prev = json.load(f)
            existing = {m['model']: m for m in results['models']}
            for m in prev.get('models', []):
                if m['model'] not in existing:
                    results['models'].append(m)
                    existing[m['model']] = m
            n_prior = len(prev.get('models', []))
            print(f'      Loaded {n_prior} prior model result(s)', flush=True)
        except Exception:
            pass

    print('[3/4] Training models...', flush=True)
    for name in args.models:
        name = name.upper()
        print(f'\n--- Model {name} ---', flush=True)
        t0 = time.time()
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'model_{name}_latest.keras')
        state_path = os.path.join(ckpt_dir, f'model_{name}_state.json')

        start_epoch, history = 0, {'accuracy': [], 'val_accuracy': []}
        if args.resume and os.path.isfile(ckpt_path) and os.path.isfile(state_path):
            candidate = keras.models.load_model(ckpt_path)
            ckpt_classes = candidate.output_shape[-1]
            if ckpt_classes == num_classes:
                model = candidate
                state = json.load(open(state_path, encoding='utf-8'))
                start_epoch = int(state['completed_epochs'])
                history = state.get('history', history)
                print(f'  Resuming from epoch {start_epoch + 1}', flush=True)
            else:
                print(f'  Ignoring checkpoint: {ckpt_classes} classes, '
                      f'need {num_classes}. Starting fresh.', flush=True)
                del candidate
                weights = None if (name == 'C' and args.no_imagenet) else 'imagenet'
                model = get_model(name, num_classes, args.img_size, weights=weights)
        else:
            weights = None if (name == 'C' and args.no_imagenet) else 'imagenet'
            model = get_model(name, num_classes, args.img_size, weights=weights)

        model.summary()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        epoch_cb = EpochCheckpoint(ckpt_path, state_path, history)
        steps_each = args.steps_per_epoch or len(train_gen)
        history = model.fit(
            train_gen,
            initial_epoch=start_epoch,
            epochs=args.epochs,
            validation_data=test_gen,
            validation_steps=args.validation_steps,
            steps_per_epoch=args.steps_per_epoch,
            callbacks=[epoch_cb, TqdmProgress(steps_each),
                       ProgressPrinter(every=100)],
            verbose=1,
        )
        history_acc = epoch_cb.history
        loss, acc = model.evaluate(test_gen, verbose=0,
                                   steps=args.validation_steps)
        elapsed = time.time() - t0
        model_path = os.path.join(args.output_dir, f'model_{name}.keras')
        model.save(model_path)

        entry = {
            'model': name,
            'test_accuracy': round(float(acc), 6),
            'test_loss': round(float(loss), 6),
            'training_time_s': round(elapsed, 1),
            'history': {
                'accuracy': [round(float(x), 5) for x in history_acc['accuracy']],
                'val_accuracy': [round(float(x), 5) for x in history_acc['val_accuracy']],
            },
            'saved_at': model_path,
        }
        results['models'].append(entry)
        print(f'Model {name} test accuracy: {acc*100:.2f}% ({elapsed:.0f}s)', flush=True)

    best = max(results['models'], key=lambda m: m['test_accuracy'])
    results['best_model'] = best['model']
    results['best_accuracy'] = best['test_accuracy']

    print('[4/4] Saving results...', flush=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = [
        '# Model Comparison Results', '',
        f"- Characters: {num_classes}",
        f"- Training samples per character: {args.num_train} original x "
        f"{args.augment_factor} augmented = {args.num_train * args.augment_factor}",
        f"- Test samples per character: ~{len(test_records) // num_classes}",
        f"- Image size: {args.img_size}x{args.img_size}", '',
        '| Model | Test Accuracy | Test Loss | Time (s) |',
        '|---|---|---|---|',
    ]
    for m in results['models']:
        lines.append(f"| {m['model']} | {m['test_accuracy']*100:.2f}% | "
                     f"{m['test_loss']:.4f} | {m['training_time_s']:.0f} |")
    lines += ['', f"**Best model: {best['model']} "
                  f"({best['test_accuracy']*100:.2f}%)**", '']
    with open(os.path.join(args.output_dir, 'RESULTS.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
