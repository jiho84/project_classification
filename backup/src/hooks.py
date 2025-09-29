"""
Kedro Hooks for implementing save_options logic with memory management
"""
import logging
import gc
from typing import Any, Dict, Optional
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline.node import Node
from kedro_mlflow.io.artifacts import MlflowArtifactDataset
import pandas as pd

from my_project.io.mlflow_memory_dataset import MlflowMemoryDataset, ConditionalSaveDataset

logger = logging.getLogger(__name__)


class SaveControllerHook:
    """
    Hook to control storage location and memory management based on save_options.
    Replaces datasets dynamically based on parameters.yml configuration.
    """
    
    def __init__(self):
        self.params = None
        self.original_catalog = None
        self.memory_tracker = {}
        
    @hook_impl
    def after_context_created(self, context) -> None:
        """Store parameters after context is created"""
        self.params = context.params
        logger.info("🔧 SaveControllerHook initialized with parameters")
        
    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        """Modify catalog based on save_options"""
        if not self.params:
            return
            
        save_options = self.params.get("save_options", {})
        logger.info("📦 Configuring catalog based on save_options...")
        
        # Get list of datasets to modify
        dataset_names = list(catalog.list())
        
        for dataset_name in dataset_names:
            # Skip non-stage datasets
            if not self._is_stage_dataset(dataset_name):
                continue
                
            stage_name = self._get_stage_name(dataset_name)
            if not stage_name:
                continue
                
            # Get save options for this stage
            stage_options = self._get_stage_save_options(stage_name)
            
            # Replace dataset if needed
            self._replace_dataset(catalog, dataset_name, stage_name, stage_options)
            
    def _replace_dataset(self, catalog: DataCatalog, dataset_name: str, 
                        stage_name: str, stage_options: Dict[str, Any]) -> None:
        """Replace dataset based on save options"""
        try:
            # Get original dataset
            original_dataset = catalog._get_dataset(dataset_name)
            
            # Skip if already a ConditionalSaveDataset
            if isinstance(original_dataset, ConditionalSaveDataset):
                return
                
            storage_location = stage_options.get("storage_location", "both")
            enabled = stage_options.get("enabled", True)
            
            # Create appropriate dataset based on configuration
            if not enabled:
                # Use MemoryDataset for disabled stages
                new_dataset = MemoryDataset()
                logger.info(f"🔄 {dataset_name}: Using MemoryDataset (stage disabled)")
                
            elif storage_location == "mlflow_only":
                # Use MlflowMemoryDataset to avoid local disk writes
                if isinstance(original_dataset, MlflowArtifactDataset):
                    # Extract artifact path from original dataset
                    artifact_path = self._extract_artifact_path(original_dataset, dataset_name)
                    dataset_type = self._get_dataset_type(dataset_name)
                    
                    new_dataset = ConditionalSaveDataset(
                        stage_name=stage_name,
                        mlflow_dataset=MlflowMemoryDataset(
                            artifact_path=artifact_path,
                            dataset_type=dataset_type
                        ),
                        save_options=self.params.get("save_options", {})
                    )
                    logger.info(f"🔄 {dataset_name}: Using MlflowMemoryDataset (mlflow_only mode)")
                else:
                    # Keep original for non-MLflow datasets
                    new_dataset = original_dataset
                    
            elif storage_location == "local_only":
                # Use local dataset only
                if isinstance(original_dataset, MlflowArtifactDataset):
                    # Extract local dataset from MlflowArtifactDataset
                    local_dataset = self._extract_local_dataset(original_dataset)
                    new_dataset = ConditionalSaveDataset(
                        stage_name=stage_name,
                        local_dataset=local_dataset,
                        save_options=self.params.get("save_options", {})
                    )
                    logger.info(f"🔄 {dataset_name}: Using local dataset only")
                else:
                    new_dataset = original_dataset
                    
            else:  # "both"
                # Keep original behavior
                new_dataset = original_dataset
                
            # Replace in catalog
            if new_dataset != original_dataset:
                catalog.add(dataset_name, new_dataset, replace=True)
                
        except Exception as e:
            logger.warning(f"Failed to replace dataset {dataset_name}: {e}")
            
    @hook_impl
    def after_node_run(self, node: Node, outputs: Dict[str, Any]) -> None:
        """Release memory after node execution"""
        # Track memory usage
        if outputs:
            for name, data in outputs.items():
                if isinstance(data, pd.DataFrame):
                    memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                    self.memory_tracker[name] = memory_usage
                    logger.debug(f"📊 {name}: {memory_usage:.2f} MB")
                    
        # Get stage name
        stage_name = None
        if node.name:
            parts = node.name.split('_')
            if len(parts) >= 2 and parts[0] == "stage":
                stage_name = f"{parts[0]}_{parts[1]}"
                
        if stage_name:
            # Check if we should release memory for this stage
            stage_options = self._get_stage_save_options(stage_name)
            if not stage_options.get("enabled", True):
                # Force garbage collection for disabled stages
                gc.collect()
                logger.debug(f"🧹 Forced garbage collection after {stage_name}")
                
    def _is_stage_dataset(self, dataset_name: str) -> bool:
        """Check if dataset is a stage dataset"""
        return (dataset_name.startswith("stage_") or 
                dataset_name in ["final_data", "final_metadata"])
        
    def _get_stage_name(self, dataset_name: str) -> Optional[str]:
        """Extract stage name from dataset name"""
        if dataset_name.startswith("stage_"):
            parts = dataset_name.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
        elif dataset_name in ["final_data", "final_metadata"]:
            return "stage_9"
        return None
        
    def _get_stage_save_options(self, stage_name: str) -> Dict[str, Any]:
        """Get save options for a specific stage"""
        if not self.params:
            return {}
            
        save_options = self.params.get("save_options", {})
        default_options = save_options.get("default", {})
        stage_overrides = save_options.get("stage_overrides", {})
        
        # Start with default options
        stage_options = default_options.copy()
        
        # Apply stage-specific overrides
        if stage_name in stage_overrides:
            stage_options.update(stage_overrides[stage_name])
            
        return stage_options
        
    def _extract_artifact_path(self, dataset: MlflowArtifactDataset, 
                              dataset_name: str) -> str:
        """Extract artifact path from MlflowArtifactDataset"""
        # Default artifact paths based on dataset name
        if dataset_name.endswith("_metadata"):
            return f"pipeline_metadata/{dataset_name}.json"
        else:
            ext = "parquet" if "stage_8" not in dataset_name else "pkl"
            return f"pipeline_data/{dataset_name}.{ext}"
            
    def _get_dataset_type(self, dataset_name: str) -> str:
        """Get dataset type based on dataset name"""
        if dataset_name.endswith("_metadata"):
            return "json.JSONDataset"
        elif "stage_8" in dataset_name or "final" in dataset_name:
            return "pickle.PickleDataset"
        else:
            return "pandas.ParquetDataset"
            
    def _extract_local_dataset(self, mlflow_dataset: MlflowArtifactDataset) -> Any:
        """Extract local dataset from MlflowArtifactDataset"""
        # MlflowArtifactDataset wraps a local dataset
        # Access the underlying dataset
        if hasattr(mlflow_dataset, "_dataset"):
            return mlflow_dataset._dataset
        return mlflow_dataset


class MemoryMonitorHook:
    """Hook to monitor memory usage throughout pipeline execution"""
    
    def __init__(self):
        self.memory_stats = {}
        
    @hook_impl
    def before_node_run(self, node: Node) -> None:
        """Record memory before node execution"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_stats[f"{node.name}_before"] = memory_mb
        
    @hook_impl
    def after_node_run(self, node: Node) -> None:
        """Record memory after node execution"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_stats[f"{node.name}_after"] = memory_mb
        
        # Calculate delta
        before = self.memory_stats.get(f"{node.name}_before", 0)
        delta = memory_mb - before
        logger.info(f"📊 Memory: {node.name} - {memory_mb:.1f} MB (Δ{delta:+.1f} MB)")
        
    @hook_impl
    def after_pipeline_run(self) -> None:
        """Report memory usage summary"""
        logger.info("="*60)
        logger.info("📊 Memory Usage Summary")
        logger.info("="*60)
        
        # Find peak memory usage
        peak_memory = max(self.memory_stats.values()) if self.memory_stats else 0
        logger.info(f"Peak memory usage: {peak_memory:.1f} MB")


# For backward compatibility
SaveOptionsHook = SaveControllerHook
StorageControlHook = SaveControllerHook