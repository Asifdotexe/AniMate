"""
Module that contains the test cases for the engine module.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src import engine


@pytest.fixture
def mock_data():
    return pd.DataFrame(
        {
            "title": ["Anime A", "Anime B", "Anime C"],
            "synopsis": ["Synopsis A", "Synopsis B", "Synopsis C"],
            "other_cols": [1, 2, 3],  # Simulating other columns
        }
    )


def test_load_data(tmp_path):
    # Create valid dummy csv
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2", encoding="utf-8")

    dtypes = {"col1": int, "col2": int}
    df = engine.load_data(str(csv_file), dtypes)

    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]


def test_vectorize_and_build_model(mock_data):
    config = {"max_features": 10, "n_neighbors": 2, "metric": "cosine"}

    knn_model, tfidf_vectorizer = engine.vectorize_and_build_model(mock_data, config)

    # Check if a column was added (stemmed_synopsis)
    assert "stemmed_synopsis" in mock_data.columns

    # Check return types (basic check)
    assert hasattr(knn_model, "kneighbors")
    assert hasattr(tfidf_vectorizer, "transform")


@patch("src.engine.preprocess_text")
def test_get_recommendations(mock_preprocess, mock_data):
    mock_preprocess.return_value = "processed query"

    # Mock Vectorizer
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = [[1, 0, 0]]  # Dummy vector

    # Mock KNN Model
    mock_knn = MagicMock()
    # Return 2 neighbors: indices 0 and 2 (Anime A and Anime C)
    mock_knn.kneighbors.return_value = (None, [[0, 2]])

    # Call get_recommendations
    # Note: We need a 'score' column or calculation logic?
    # Looking at engine.py, get_recommendations sorts by 'score'.
    # But wait, engine.py doesn't seem to calculate a score in get_recommendations logic shown in previous turn?
    # Checking source:
    # final_recommendations = filtered_recommendations.sort_values(by="score", ascending=False).head(top_n)
    # The 'score' column MUST exist in the data frame passed in or be calculated?
    # Wait, the engine.py I read earlier relies on 'score' column existing in the dataframe?
    # Or is duplicate logic missing?
    # Let's assume for now 'score' needs to be in data or we'll get a KeyError.
    # Actually, standard KNN usually returns distances which act as scores.
    # engine.py:78: _, indices = knn_model.kneighbors(...)
    # It ignores distances!
    # And then lines 92-94 sort by 'score'.
    # This implies 'score' is a pre-existing column in 'data', perhaps popularity or rating?
    # I will add 'score' to mock_data config.

    mock_data["score"] = [9.0, 8.0, 7.0]

    recs = engine.get_recommendations(
        "query", mock_vectorizer, mock_knn, mock_data, top_n=2
    )

    assert len(recs) <= 2
    # Should contain Anime A (index 0) and Anime C (index 2)
    assert "Anime A" in recs["title"].values
    assert "Anime C" in recs["title"].values

    # Verify sorting: Anime A (9.0) > Anime C (7.0)
    assert recs.iloc[0]["title"] == "Anime A"
