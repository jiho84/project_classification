"""MLflow context diagnostic script."""
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mlflow

    # Set tracking URI
    mlflow.set_tracking_uri("file:///home/user/projects/kedro_project/account-tax/mlruns")

    # Check active run
    active = mlflow.active_run()
    logger.info(f"Active run: {active}")

    if active:
        logger.info(f"  Run ID: {active.info.run_id}")
        logger.info(f"  Run name: {active.info.run_name}")
        logger.info(f"  Status: {active.info.status}")
        logger.info(f"  Artifact URI: {active.info.artifact_uri}")
    else:
        logger.info("No active MLflow run found")

    # List all runs
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment_by_name("account_tax_experiment")

    if experiment:
        logger.info(f"\nExperiment ID: {experiment.experiment_id}")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        logger.info(f"\nFound {len(runs)} runs:")
        for run in runs:
            logger.info(f"  - {run.info.run_id[:8]}... | {run.info.run_name} | Status: {run.info.status} | End: {run.info.end_time}")

            # Check artifacts
            artifacts = client.list_artifacts(run.info.run_id)
            if artifacts:
                logger.info(f"    Artifacts: {[a.path for a in artifacts]}")
            else:
                logger.info(f"    Artifacts: None")

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
