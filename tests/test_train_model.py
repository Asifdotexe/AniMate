"""
Module for testing the training script.
"""

from unittest.mock import MagicMock, mock_open, patch

from src.pipeline.train import train


@patch("os.makedirs")
@patch("joblib.dump")
@patch("src.pipeline.train.NearestNeighbors")
@patch("src.pipeline.train.TfidfVectorizer")
@patch("pandas.read_csv")
@patch("os.path.exists")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="model:\n  max_features: 100\n  n_neighbors: 5\n",
)
@patch("yaml.safe_load")
def test_train_model(
    mock_yaml,
    mock_open_file,
    mock_exists,
    mock_read_csv,
    mock_tfidf,
    mock_knn,
    mock_dump,
    mock_makedirs,
):
    """
    Test the training script execution, verifying that data loading, model training,
    and artifact saving are orchestrated correctly.
    """
    # Setup mocks
    mock_yaml.return_value = {
        "model": {"max_features": 100, "n_neighbors": 5, "metric": "cosine"}
    }
    mock_exists.return_value = True  # Processed data exists

    # Mock DataFrame
    mock_df = MagicMock()
    mock_df.columns = ["Title", "Synopsis"]
    # Mock apply for stemming
    mock_df.__getitem__.return_value = mock_df  # simplistic
    mock_read_csv.return_value = mock_df

    # Mock Vectorizer and KNN
    mock_vectorizer_instance = MagicMock()
    mock_tfidf.return_value = mock_vectorizer_instance
    mock_vectorizer_instance.fit_transform.return_value = "tfidf_matrix"

    mock_knn_instance = MagicMock()
    mock_knn.return_value = mock_knn_instance

    # Run train function
    train()

    # Verify logic
    mock_read_csv.assert_called_once()
    mock_tfidf.assert_called_once()
    mock_knn.assert_called_once()

    # Check if artifacts were saved.
    assert mock_dump.call_count == 2
    mock_df.to_pickle.assert_called_once()
    mock_makedirs.assert_called()
