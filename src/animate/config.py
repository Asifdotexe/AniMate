from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Define paths for data
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "AnimeData_300724.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "AnimeData_300724_processed.csv"

# Define paths for models
MODELS_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.joblib"

# Model training parameters
# 5 recommendations + the item itself
N_NEIGHBORS = 6
MAX_FEATURES = 5000
