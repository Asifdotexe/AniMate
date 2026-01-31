"""
This is the module that contains the test cases for the engine module.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.inference import engine


@pytest.fixture
def mock_data():
    """Create a mock DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": ["Anime A", "Anime B", "Anime C"],
            "synopsis": ["Synopsis A", "Synopsis B", "Synopsis C"],
            # Simulating other columns
            "other_cols": [1, 2, 3],
        }
    )


@patch("src.inference.engine.preprocess_text")
def test_get_recommendations(mock_preprocess, mock_data):
    """Test getting recommendations based on a query."""
    mock_preprocess.return_value = "processed query"

    # Mock Vectorizer
    mock_vectorizer = MagicMock()
    # Dummy vector
    mock_vectorizer.transform.return_value = [[1, 0, 0]]

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


@patch("pathlib.Path.exists")
@patch("joblib.load")
def test_load_models_success(mock_joblib_load, mock_exists):
    """Test successful loading of models."""
    mock_exists.return_value = True
    mock_joblib_load.side_effect = ["knn_model", "tfidf_vectorizer"]

    knn, vectorizer = engine.load_models("models")

    assert knn == "knn_model"
    assert vectorizer == "tfidf_vectorizer"
    assert mock_joblib_load.call_count == 2


@patch("pathlib.Path.exists")
def test_load_models_file_not_found(mock_exists):
    """Test that FileNotFoundError is raised when models are missing."""
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError):
        engine.load_models("models")


@patch("pathlib.Path.exists")
@patch("pandas.read_pickle")
def test_load_processed_data(mock_read_pickle, mock_exists):
    """Test loading of processed data."""
    mock_exists.return_value = True
    mock_read_pickle.return_value = pd.DataFrame({"col": [1, 2]})

    df = engine.load_processed_data("models")

    assert not df.empty
    assert list(df.columns) == ["col"]
