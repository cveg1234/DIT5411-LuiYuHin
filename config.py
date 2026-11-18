"""Central configuration for the DIT5411 character-recognition project."""

# Dataset
DATA_DIR = "data/cleaned_data"      # folder with one subfolder per character
NUM_TRAIN_PER_CHAR = 40             # first 40 samples = training set
AUGMENT_FACTOR = 5                  # 40 x 5 = 200 training samples per character

# Input
IMG_SIZE = 128                      # resize 300x300 source images to 128x128
GRAYSCALE = False                   # models consume RGB (repeat grayscale channel)

# Training
BATCH_SIZE = 128
EPOCHS = 15
VALIDATION_SPLIT = 0.0              # we use the assignment's own test split instead

# Output
OUTPUT_DIR = "results"
LABEL_MAP_FILE = "label_map.json"
RESULTS_FILE = "results.json"
