"""Test script to manually upload model artifacts to MLflow."""
import json
import logging
from pathlib import Path

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RUN_ID = "bdaf1bb13de440388bd4d4b1bf7aac0f"
CHECKPOINTS_DIR = Path("/home/user/projects/kedro_project/account-tax/data/06_models/checkpoints")
MLFLOW_TRACKING_URI = "file:///home/user/projects/kedro_project/account-tax/mlruns"

def main():
    """Upload model artifacts to MLflow run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Get the existing run
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(RUN_ID)
    logger.info(f"Found MLflow run: {run.info.run_name} (status: {run.info.status})")

    # Resume the run to add artifacts
    with mlflow.start_run(run_id=RUN_ID):
        logger.info(f"Resumed run {RUN_ID}")

        # Log final model directory
        if CHECKPOINTS_DIR.exists():
            logger.info(f"Logging final/best model from {CHECKPOINTS_DIR}")
            mlflow.log_artifacts(str(CHECKPOINTS_DIR), artifact_path="final_model")
            logger.info("✓ Successfully logged final_model artifacts")
        else:
            logger.warning(f"Checkpoints directory not found: {CHECKPOINTS_DIR}")

        # Find and log best checkpoint
        trainer_state_path = CHECKPOINTS_DIR / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.load(f)

                best_checkpoint = trainer_state.get("best_model_checkpoint")
                if best_checkpoint:
                    best_ckpt_path = Path(best_checkpoint)
                    if best_ckpt_path.exists() and best_ckpt_path != CHECKPOINTS_DIR:
                        logger.info(f"Logging best checkpoint from {best_ckpt_path}")
                        mlflow.log_artifacts(str(best_ckpt_path), artifact_path="best_checkpoint")
                        logger.info("✓ Successfully logged best_checkpoint artifacts")
                    else:
                        logger.info(f"Best checkpoint same as final model or not found: {best_ckpt_path}")
                else:
                    logger.info("No best_model_checkpoint found in trainer_state.json")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error reading trainer_state.json: {e}")
        else:
            logger.warning(f"trainer_state.json not found: {trainer_state_path}")

    logger.info("=" * 80)
    logger.info("SUCCESS: MLflow artifacts upload test completed!")
    logger.info("=" * 80)

    # Verify artifacts were uploaded
    artifacts = client.list_artifacts(RUN_ID)
    logger.info(f"Artifacts in run {RUN_ID}:")
    for artifact in artifacts:
        logger.info(f"  - {artifact.path} ({'dir' if artifact.is_dir else 'file'})")

if __name__ == "__main__":
    main()
