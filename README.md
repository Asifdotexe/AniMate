# RecommendationHaki (見聞色)

> **"Predicting your next favorite anime before you even know it."**

**RecommendationHaki** is an intelligent anime discovery engine designed to cure the modern plague of decision paralysis.

---

## The Inspiration

The project is named after **Kenbunshoku Haki (Observation Haki)** from *One Piece*; a spiritual energy that grants the user a "sixth sense" to gauge the strength of others, sense their presence, and most importantly, **predict their future moves**.

Just as a Haki user can sense an incoming attack, **RecommendationHaki** analyzes the hidden patterns in your viewing history to sense exactly what you're craving next. It cuts through the noise of thousands of mediocre shows to find the "strongest" matches for your specific taste.

## The Tech Behind the Haki

Under the hood, this isn't magic; it's high-dimensional mathematics.

*   **Vector Embeddings**: We convert anime synopses, genres, and themes into high-dimensional vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency).
*   **KNN Algorithms**: We use **K-Nearest Neighbors** to map the "distance" between shows, finding the hidden narrative connections that simple genre tags miss.
*   **Live Data**: Powered by the **Jikan API** (MyAnimeList), ensuring the database is always current.

> **Don't just watch anime. Sense it.**

---

## Getting Started

### Prerequisites
*   Python 3.11+
*   Poetry (Recommended) or pip

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Asifdotexe/RecommendationHaki.git
    cd RecommendationHaki
    ```

2.  **Install Dependencies**
    ```bash
    poetry install
    # OR
    pip install -r requirements.txt
    ```

3.  **Run the Training Pipeline**
    (Optional, artifacts are included but you can regenerate them)
    ```bash
    poetry run python src/pipeline/training_pipeline.py
    ```

4.  **Activate Haki (Run the App)**
    ```bash
    poetry run streamlit run app/main.py
    ```

## 📂 Project Structure

```
RecommendationHaki/
├── app/                  # Streamlit Application
│   ├── main.py          # Entry point
│   └── assets/          # Static assets (CSS, Images)
├── src/                  # Core Logic
│   ├── components/      # Data Ingestion, Transformation, Trainer
│   ├── pipeline/        # Inference Engine, Training Pipeline
│   ├── config.py        # Configuration Parser
│   └── logger.py        # Logging Utility
├── artifacts/            # Model & Data Artifacts (Joblib, PKL)
├── config.yaml           # Centralized Configuration
└── pyproject.toml        # Dependencies
```

## 🤝 Contribution

Feel free to fork this repository and submit pull requests. To train your own Haki, tweak the `config.yaml` hyperparameters!

---
*Built with ❤️ and Haki by [Asif Sayyed](https://github.com/Asifdotexe)*
