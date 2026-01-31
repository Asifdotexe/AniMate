"""
Training pipeline orchestration script.
Runs the full data pipeline: Ingestion -> Transformation -> Training.
"""

import sys
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.components import ingestion, trainer, transformation
from src.logger import setup_logging

logger = setup_logging("training_pipeline")


def run_pipeline():
    """
    Execute the full training pipeline.
    """
    logger.info("Starting Training Pipeline...")

    try:
        logger.info("Data Ingestion (1/3)")
        ingestion.main()

        logger.info("Data Transformation (2/3)")
        transformation.main()

        logger.info("Model Training (3/3)")
        trainer.train()

        logger.info("Training Pipeline Completed Successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
