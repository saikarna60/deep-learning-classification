"""Central configuration for the deep-learning classification project."""
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

# Image classifier settings
IMG_MODEL_PATH = MODEL_DIR / "image_classifier.pt"
IMG_BATCH_SIZE = 32
IMG_EPOCHS = 5
IMG_LR = 1e-3

# Text classifier settings
TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_MODEL_PATH = MODEL_DIR / "text_classifier.pt"
TEXT_BATCH_SIZE = 16
TEXT_EPOCHS = 3
TEXT_LR = 2e-5
TEXT_MAX_LEN = 128
