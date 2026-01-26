"""
Module that contains the test cases for the train_model module.
"""

import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

from train_model import train

# Ensure scripts can be imported
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)


@patch("os.makedirs")
@patch("joblib.dump")
@patch("src.engine.vectorize_and_build_model")
@patch("src.engine.load_data")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="data:\n  file_path: path\n  dtypes: {}\nmodel: {}",
)
@patch("yaml.safe_load")
def test_train_model(
    mock_yaml, mock_file, mock_load_data, mock_vectorize, mock_dump, mock_makedirs
):
    # Setup mocks
    mock_yaml.return_value = {
        "data": {"file_path": "data.csv", "dtypes": {}},
        "model": {},
    }

    mock_df = MagicMock()
    mock_load_data.return_value = mock_df

    mock_knn = MagicMock()
    mock_vectorizer = MagicMock()
    mock_vectorize.return_value = (mock_knn, mock_vectorizer)

    # Run train function
    train()

    # Verify logic
    mock_load_data.assert_called_once()
    mock_vectorize.assert_called_once_with(mock_df, {})

    # Check if artifacts were saved.
    assert mock_dump.call_count == 2
    mock_df.to_pickle.assert_called_once()
    mock_makedirs.assert_called()
