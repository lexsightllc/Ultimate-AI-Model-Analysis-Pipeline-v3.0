"""Ultimate AI Model Analysis Pipeline package."""
from .advanced_pipeline import (
    AnalysisConfig as AdvancedAnalysisConfig,
    CalibrationMathematics,
    ManifoldDimensionalityReducer,
    UltimateModelAnalysisPipeline,
    create_synthetic_classification_data,
    demonstrate_competition_submission,
    demonstrate_pipeline_comprehensive,
    optimize_calibration_parameters,
)
from .config import AnalysisConfig, load_default_config
from .evaluation import EvaluationSummary, evaluate_prediction_file
from .pipeline import AnalysisPipeline, PipelineResult, run_pipeline
from .multilabel.pipeline import (
    AnalysisConfig as MultilabelAnalysisConfig,
    UltimateMultilabelPipeline,
    load_config as load_multilabel_config,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisPipeline",
    "PipelineResult",
    "load_default_config",
    "run_pipeline",
    "evaluate_prediction_file",
    "EvaluationSummary",
    "UltimateMultilabelPipeline",
    "MultilabelAnalysisConfig",
    "load_multilabel_config",
    "UltimateModelAnalysisPipeline",
    "AdvancedAnalysisConfig",
    "CalibrationMathematics",
    "ManifoldDimensionalityReducer",
    "optimize_calibration_parameters",
    "create_synthetic_classification_data",
    "demonstrate_pipeline_comprehensive",
    "demonstrate_competition_submission",
]
