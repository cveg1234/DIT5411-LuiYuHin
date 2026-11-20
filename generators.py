"""On-the-fly data generators. No 2.6M-image folder is written to disk.

Training generator yields 40 originals x 5 variants = 200 images per character.
Testing generator yields the untouched held-out samples.
"""
import math
import numpy as np
from tensorflow.keras.utils import Sequence

from augmentation import load_gray, affine_augment


class HandwritingSequence(Sequence):
    def __init__(self, records, batch_size, num_classes, img_size,
                 factor=1, shuffle=True, seed=42, rgb=True):
        """
        records: list of (image_path, label_index) for ORIGINAL samples.
        factor:  how many versions to yield per original (5 for training, 1 for test).
        """
        super().__init__()  # required by Keras Sequence; silences PyDataset warning
        self.records = records
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_size = img_size
        self.factor = factor
        self.shuffle = shuffle
        self.rgb = rgb
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.indices = np.arange(len(records) * factor)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.epoch += 1

    def __getitem__(self, index):
        batch_idx = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]
        X = np.empty((len(batch_idx), self.img_size, self.img_size, 3 if self.rgb else 1),
                     dtype='float32')
        y = np.empty((len(batch_idx),), dtype='int32')

        for j, global_i in enumerate(batch_idx):
            record_i = int(global_i) // self.factor
            variant = int(global_i) % self.factor
            path, label = self.records[record_i]
            img = load_gray(path, self.img_size)
            if variant > 0:
                img = affine_augment(img, self.rng)
            arr = img.astype('float32') / 255.0
            if self.rgb:
                arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
            else:
                arr = arr[..., np.newaxis]
            X[j] = arr
            y[j] = label

        return X, np.eye(self.num_classes, dtype='float32')[y]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
