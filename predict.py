"""Predict which Chinese character is in a handwritten image.

Usage:
    python predict.py --image sample.png --model results/model_A.keras
"""
import argparse
import json
import numpy as np
from tensorflow import keras

from augmentation import load_gray
from dataset import load_label_map


def predict(image_path, model_path, label_map_path, img_size=128, top_k=5):
    model = keras.models.load_model(model_path)
    chars = load_label_map(label_map_path)
    img = load_gray(image_path, img_size).astype('float32') / 255.0
    x = np.repeat(img[..., np.newaxis], 3, axis=-1)[np.newaxis, ...]
    probs = model.predict(x, verbose=0)[0]
    top = np.argsort(probs)[-top_k:][::-1]
    return [(chars[i], float(probs[i])) for i in top]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--label-map', default='results/label_map.json')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--top-k', type=int, default=5)
    a = p.parse_args()
    for ch, prob in predict(a.image, a.model, a.label_map, a.img_size, a.top_k):
        print(f'{ch}  {prob*100:.2f}%')
