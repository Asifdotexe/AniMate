# Data Pipeline Instructions

This directory contains the sequential data pipeline for the AniMate project.

## 1. Data Collection
Scrape data from MyAnimeList and save it to `data/raw`.

```bash
python src/pipeline/collect.py
```

## 2. Data Processing
Clean and preprocess the raw data, merging it into a single dataset saved to `data/processed`.

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
