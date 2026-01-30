# Data Pipeline Instructions

This directory contains the sequential data pipeline for the AniMate project.

## 1. Data Collection (Upsert)
Fetch popular anime data using the Jikan API (v4). 
This script implements an **Upsert** strategy:
- Updates existing anime in `data/raw/anime_master_db.csv`
- Appends new unique anime found in the API response.

```bash
python src/pipeline/collect_api.py
```

## 2. Data Processing
Reads the persistent `data/raw/anime_master_db.csv`, cleans it, and applies text preprocessing.
The result is saved to `data/processed/anime_data_processed.csv`.

```bash
python src/pipeline/process.py
```

## 3. Model Training
Train the recommendation model and generate vector artifacts.
- Saves model to `models/`
- Saves vectorizer and processed data pickle to `models/`

```bash
python src/pipeline/train.py
```
