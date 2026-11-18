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
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
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

    print('[3/4] Training models...', flush=True)
    for name in args.models:
        name = name.upper()
        print(f'\n--- Model {name} ---', flush=True)
        t0 = time.time()
        weights = None if (name == 'C' and args.no_imagenet) else 'imagenet'
        model = get_model(name, num_classes, args.img_size, weights=weights)
        model.summary()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        history = model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=test_gen,
            steps_per_epoch=args.steps_per_epoch,
            verbose=1,
        )
        loss, acc = model.evaluate(test_gen, verbose=0)
        elapsed = time.time() - t0
        model_path = os.path.join(args.output_dir, f'model_{name}.keras')
        model.save(model_path)

        entry = {
            'model': name,
            'test_accuracy': round(float(acc), 6),
            'test_loss': round(float(loss), 6),
            'training_time_s': round(elapsed, 1),
            'history': {
                'accuracy': [round(float(x), 5) for x in history.history['accuracy']],
                'val_accuracy': [round(float(x), 5) for x in history.history['val_accuracy']],
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
