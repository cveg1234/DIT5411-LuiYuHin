# Model Comparison Results

- Characters: 13065
- Training samples per character: 40 original x 5 augmented = 200
- Test samples per character: ~12
- Image size: 128x128

| Model | Test Accuracy | Test Loss | Time (s) |
|---|---|---|---|
| A | 85.74% | 0.5363 | 6684 |
| B | 0.07% | 22.2270 | 6571 |
| C | 0.01% | 9.4775 | 6008 |

**Best model: A (85.74%)**

Note: Model A improved from 71.99% (after 1 epoch) to 82.99% and then 85.74%
after further epochs of continued training, confirming the extra epochs helped.
Model B was also continued for a second epoch but stayed near random (0.05% -> 0.07%,
loss 22.7 -> 22.2), confirming it had collapsed and could not be recovered.
Model C (MLP baseline) also failed to learn (0.01%, loss ~ln(13065)=9.48).
