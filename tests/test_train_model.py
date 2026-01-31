"""
Module for testing the training script.
"""

from unittest.mock import MagicMock, patch

from src.components.trainer import train


@patch("src.components.trainer.config")
@patch("pathlib.Path.mkdir")
@patch("joblib.dump")
@patch("src.components.trainer.NearestNeighbors")
@patch("src.components.trainer.TfidfVectorizer")
@patch("pandas.read_csv")
@patch("pathlib.Path.exists")  # Patch Path.exists for processed data check
def test_train_model(
    mock_exists,
    mock_read_csv,
    mock_tfidf,
    mock_knn,
    mock_dump,
    mock_mkdir,
    mock_config,
):
    """
    Test the training script execution, verifying that data loading, model training,
    and artifact saving are orchestrated correctly.
    """
    # Setup mocks
    mock_config.model.vectorizer_max_features = 100
    mock_config.model.top_k_recommendations = 5  # Used as n_neighbors
    mock_config.paths.processed_data = "data/processed/cleaned_anime_data.csv"
    mock_config.paths.knn_model = "artifacts/knn_model.joblib"
    mock_config.paths.vectorizer = "artifacts/vectorizer.joblib"
    mock_config.paths.vector_embeddings = "artifacts/vector_embeddings.pkl"

    mock_exists.return_value = True  # Processed data exists

    # Mock DataFrame
    mock_df = MagicMock()
    mock_df.columns = [
        "Title",
        "Synopsis",
        "stemmed_synopsis",
    ]  # ensure 'stemmed_synopsis' exists to avoid calling preprocess
    # Mock apply for stemming (not called if 'stemmed_synopsis' exists, but if it did)
    mock_df.__getitem__.return_value = mock_df
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
    assert mock_mkdir.call_count >= 1  # We mkdir multiple times
