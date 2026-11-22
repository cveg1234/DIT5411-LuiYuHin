# Model Comparison Results

- Characters: 13065
- Training samples per character: 40 original x 5 augmented = 200
- Test samples per character: ~12
- Image size: 128x128

| Model | Test Accuracy | Test Loss | Time (s) |
|---|---|---|---|
| A | 82.99% | 0.6432 | 6392 |
| B | 0.07% | 22.2270 | 6571 |

**Best model: A (82.99%)**

Note: Model A improved from 71.99% (after 1 epoch) to 82.99% after a second
epoch of continued training, confirming the extra epoch helped.
