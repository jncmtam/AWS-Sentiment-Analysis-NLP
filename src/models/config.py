import torch

class Config:
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 2
    PATIENCE = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "models/sentiment_model"

config = Config()