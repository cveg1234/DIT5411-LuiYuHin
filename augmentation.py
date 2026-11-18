"""Data augmentation using OpenCV affine transforms.

The assignment requires creating at least 200 training samples per character
from only 40 originals, using rotation, shearing and scaling.

We generate 5 versions of every original (1 original + 4 augmented),
so 40 x 5 = 200 training samples per character.
"""
import cv2
import numpy as np

RANDOM_STATE = 12345
_rng = np.random.default_rng(RANDOM_STATE)


def load_gray(path, size):
    """Read an image, convert to grayscale, resize to (size, size).

    Uses imdecode because cv2.imread cannot open non-ASCII paths on Windows.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def affine_augment(img, rng=None):
    """Apply a random rotation + shear + scale + translation.

    All transforms are combined into ONE affine matrix for speed.
    """
    if rng is None:
        rng = _rng
    h, w = img.shape
    center = (w / 2.0, h / 2.0)

    angle = rng.uniform(-10, 10)          # rotation
    shear = rng.uniform(-0.10, 0.10)      # shearing
    scale = rng.uniform(0.90, 1.10)       # scaling
    tx = rng.uniform(-0.05, 0.05) * w     # translation
    ty = rng.uniform(-0.05, 0.05) * h

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 1] += shear                        # add horizontal shear
    M[0, 2] += tx
    M[1, 2] += ty

    out = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,                    # white background (ink = dark)
    )
    return out


def augment_batch(images, labels, factor, size, rgb=True, rng=None):
    """Expand a batch of grayscale images into factor x more samples.

    images: list/array of grayscale uint8 (size,size) arrays
    returns: X (float32, N*factor, size,size,channels),
             y (int32 labels repeated)
    """
    if rng is None:
        rng = _rng
    X, y = [], []
    for img, label in zip(images, labels):
        for k in range(factor):
            if k == 0:
                out = img
            else:
                out = affine_augment(img, rng)
            arr = out.astype('float32') / 255.0
            X.append(arr)
            y.append(label)
    X = np.asarray(X, dtype='float32')
    if rgb:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    else:
        X = X[..., np.newaxis]
    return X, np.asarray(y, dtype='int32')


def augment_one(path, size, variant, rgb=True, rng=None):
    """Load one image and produce a specific variant (0=original, 1..=augmented)."""
    if rng is None:
        rng = _rng
    img = load_gray(path, size)
    if variant > 0:
        img = affine_augment(img, rng)
    arr = img.astype('float32') / 255.0
    if rgb:
        arr = np.repeat(arr[..., np.newaxis], 3, axis=-1)
    else:
        arr = arr[..., np.newaxis]
    return arr
