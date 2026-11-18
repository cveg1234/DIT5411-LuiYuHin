"""Dataset loading and train/test splitting for the DIT5411 assignment.

Assignment rule: for every character, the FIRST 40 samples (sorted by filename)
are the training set and the remaining samples are the testing set.

The AI-FREE-Team zip stores each character in a numeric folder
(e.g. cleaned_data/10000/颮_0.png), so the real character is read from the
filenames rather than the folder name.
"""
import json
import os
import re


def natural_key(filename):
    """Sort 'a2.png' before 'a10.png' (natural numeric ordering)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


def resolve_char_dir(data_dir, folder):
    """Return (image_dir, character) for one top-level folder.

    Normal layout:  cleaned_data/10000/颮_0.png
    Symbol layout:  cleaned_data/10049/└/_0.png   (extra nested folder)
    """
    d = os.path.join(data_dir, folder)
    imgs = [f for f in os.listdir(d)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if imgs:
        first = sorted(imgs, key=natural_key)[0]
        return d, re.sub(r'_\d+\.(png|jpg|jpeg)$', '', first,
                         flags=re.IGNORECASE)
    # Nested layout: one subdirectory holding the images
    subs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    if len(subs) == 1:
        sub = os.path.join(d, subs[0])
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg'))
               for f in os.listdir(sub)):
            return sub, subs[0]
    return None, None


def char_from_folder(char_dir):
    """Read the character label from the first image filename in a folder."""
    files = [f for f in os.listdir(char_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None
    first = sorted(files, key=natural_key)[0]
    return re.sub(r'_\d+\.(png|jpg|jpeg)$', '', first, flags=re.IGNORECASE)


def build_label_map(data_dir):
    """Return (chars, char2idx). chars is a stable sorted list of characters."""
    folders = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    pairs = []
    for f in folders:
        _d, ch = resolve_char_dir(data_dir, f)
        if ch:
            pairs.append((f, ch))
    if not pairs:
        raise RuntimeError(
            f"No character folders found in {data_dir}. "
            "Did you extract with Big5 decoding (see extract_dataset.py)?"
        )
    pairs.sort(key=lambda p: (int(p[0]) if p[0].isdigit() else 0, p[0]))
    chars = [c for _f, c in pairs]
    char2idx = {c: i for i, c in enumerate(chars)}
    return chars, char2idx


def list_samples(char_dir):
    """Return naturally sorted image filenames in one character folder."""
    files = [f for f in os.listdir(char_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(files, key=natural_key)


def split_dataset(data_dir, num_train=40, max_chars=None):
    """Split every character's samples into train (first N) and test (rest).

    Returns dict: {char: {'train': [abs paths], 'test': [abs paths]}}
    """
    chars, _ = build_label_map(data_dir)
    if max_chars:
        chars = chars[:max_chars]

    folder_of = {}
    for f in os.listdir(data_dir):
        d, ch = resolve_char_dir(data_dir, f)
        if d and ch:
            folder_of.setdefault(ch, d)

    splits = {}
    for ch in chars:
        char_dir = folder_of.get(ch)
        if not char_dir:
            continue
        samples = list_samples(char_dir)
        if len(samples) < 2:
            continue
        n_train = min(num_train, len(samples) - 1)  # always keep >=1 test sample
        paths = [os.path.join(char_dir, s) for s in samples]
        splits[ch] = {'train': paths[:n_train], 'test': paths[n_train:]}
    return splits


def save_label_map(chars, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chars, f, ensure_ascii=False, indent=2)


def load_label_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/cleaned_data')
    p.add_argument('--out', default='results/label_map.json')
    p.add_argument('--num-train', type=int, default=40)
    args = p.parse_args()

    splits = split_dataset(args.data_dir, args.num_train)
    chars = sorted(splits.keys(), key=lambda x: x.encode('utf-8'))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    save_label_map(chars, args.out)
    train_total = sum(len(v['train']) for v in splits.values())
    test_total = sum(len(v['test']) for v in splits.values())
    print(f"Characters: {len(chars)}")
    print(f"Training samples (originals): {train_total}")
    print(f"Testing samples: {test_total}")
    print(f"Label map saved to {args.out}")
