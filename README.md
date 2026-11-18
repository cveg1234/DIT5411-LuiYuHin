# DIT5411-LuiYuHin — Traditional Chinese Handwriting Recognition

THEi DIT5411 Machine Learning Assignment — Character Recognition.

A TensorFlow / Keras convolutional neural network that recognises handwritten
Traditional Chinese characters from the 13,065-class AI-FREE-Team dataset.

---

## 1. Objective

Build a Python program with the TensorFlow framework that recognises Chinese
handwritten images. The dataset contains **13,065 different characters** with an
average of 50 samples each. The first 40 samples of each character are the
training set and the remainder is the testing set. Because 40 samples is too
few, images are augmented to at least **200 training samples per character**.

---

## 2. Dataset

The full 13,065-character dataset is distributed as split zip files in a
separate repository.

```bash
git clone https://github.com/chenkenanalytic/handwritting_data_all.git
cd handwritting_data_all
cat all_data.zip* > all_data.zip
unzip -O big5 all_data.zip -d data/
```

The `-O big5` flag is essential: without it the character folder names become
mojibake. After extraction the images live in `data/cleaned_data/<character>/*.png`
(300x300 pixels, 684,677 images total).

Source dataset: [AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)

---

## 3. Method

### 3.1 Train / test split

`dataset.py` naturally sorts the filenames in each character folder. The first
40 become the training set, the rest the testing set. Characters with fewer than
50 samples keep at least one test image. A stable label map (character → index
0…13,064) is saved to `results/label_map.json`.

### 3.2 Augmented image creation

`augmentation.py` uses OpenCV affine transforms. One affine matrix combines:

| Transform | Range |
|---|---|
| Rotation | ±10 degrees |
| Shearing | ±0.10 |
| Scaling | 0.90 – 1.10 |
| Translation | ±5% of width / height |

`cv2.getRotationMatrix2D` builds the rotation/scale matrix, horizontal shear is
added to the matrix, and `cv2.warpAffine` applies it with a white border
(ink is dark on white paper). For every original sample we keep 1 original and
generate 4 randomly transformed versions: **40 × 5 = 200 training samples per
character**. The test set is never augmented.

`generators.py` yields augmented batches on the fly, so ~2.6 million augmented
images never need to be written to disk. Images are resized to 128×128 and
normalised to [0, 1].

### 3.3 Models

All three models end with a `Dense(13_065, activation='softmax')` output layer.

| Model | Architecture |
|---|---|
| **A — Simple CNN** | Conv2D(32) → Pool → Conv2D(64) → Pool → Flatten → Dense(512) → Dropout(0.3) → Output |
| **B — Deep CNN** | Four `Conv→BN→Conv→BN→Pool` blocks (32/64/128/256 filters) with increasing dropout, GlobalAveragePooling → Dense(1024) → Dropout(0.4) → Output |
| **C — Transfer learning** | MobileNetV2 (ImageNet weights, frozen) → GlobalAveragePooling → Dropout(0.3) → Dense(1024) → Dropout(0.3) → Output |

All models use Adam (lr 1e-3) and categorical cross-entropy.

---

## 4. Results

`train.py` writes `results/results.json` and `results/RESULTS.md`; the full
comparison table is generated there after the real dataset has been trained.
The best model is selected by test accuracy.

> Note: TensorFlow ≥2.11 has no native GPU support on Windows. For the full
> 13,065-class run use **Google Colab (T4 GPU)** or **WSL2** with CUDA. Native
> Windows CPU training is only suitable for the small pilot run.

---

## 5. How to run

```bash
# Python 3.11 recommended
python -m venv .venv
.venv/Scripts/activate            # Windows
pip install -r requirements.txt

# Quick pilot run (20 characters, 2 epochs, models A and B)
python train.py --num-classes 20 --epochs 2 --models A B --no-imagenet

# Full assignment run
python train.py --epochs 15

# Predict a single image
python predict.py --image sample.png --model results/model_A.keras
```

Google Colab is also supported. Use **`notebooks/DIT5411_Drive.ipynb`** — it mounts
Google Drive and stores the dataset, checkpoints and results under
`MyDrive/DIT5411/`, so they survive runtime resets and GPU-quota cutoffs. Enable
a GPU (`Runtime → Change runtime type → T4 GPU`), run the setup cell once, then
run the training cell. If the session dies, run the setup cell again (remounts
Drive) and the training cell with `--resume` to continue from the last saved epoch.

---

## 6. Repository structure

```
DIT5411-LuiYuHin/
├── config.py            # default hyperparameters
├── dataset.py           # label map and first-40 / rest split
├── augmentation.py      # OpenCV affine augmentation
├── generators.py        # on-the-fly Keras Sequence generators
├── models.py            # models A, B and C
├── train.py             # train + compare + save results
├── predict.py           # single-image inference
├── requirements.txt
├── notebooks/
│   └── DIT5411_Colab.ipynb
└── results/             # generated: label_map.json, results.json, RESULTS.md, models
```

---

## 7. References

1. AI.FREE Team, *Traditional Chinese Handwriting Dataset* (2020). [GitHub](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset)
2. C. Chen, *handwritting_data_all* (full 13,065-character dataset). [GitHub](https://github.com/chenkenanalytic/handwritting_data_all)
3. D. Jain, *Introduction to OpenCV*. [GitHub](https://github.com/darshitajain/Introduction-to-OpenCV)
4. kimanalytics, *Handwritten Digit Recognition using Keras and TensorFlow*. [GitHub](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow)
5. TensorFlow/Keras documentation. [tensorflow.org](https://www.tensorflow.org/)

## License / attribution

The dataset is licensed CC BY-NC-SA 4.0 and was developed by AI.FREE Team from
STUST EECS Chinese MNIST (total set). Cite the source when reusing it.
