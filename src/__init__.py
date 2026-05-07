# __init__.py
# Project: Deadlock Predictor - Deadlock-1.0
# Makes src a Python package
# Author: Hari Krishnan M R
# Date: May 2026

from src.preprocessing import (
    load_trace_parts,
    clean_trace,
    create_labels,
    get_missing_summary,
    COLUMNS,
    EVENT_TYPES,
    FAILURE_EVENTS,
    NORMAL_EVENTS,
    TERMINAL_EVENTS
)

from src.features import (
    calculate_scheduling_delay,
    calculate_execution_duration,
    calculate_machine_failure_rate,
    calculate_concurrency_density,
    calculate_resource_overcommit,
    calculate_priority_gap,
    build_feature_matrix
)

from src.models import (
    prepare_data,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    predict,
    save_model,
    load_model,
    get_feature_importance,
    RAW_FEATURES,
    NOVEL_FEATURES,
    ALL_FEATURES
)

from src.evaluate import (
    evaluate_model,
    run_ablation_study,
    run_cross_validation,
    run_paired_ttest,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_ablation
)

__version__ = "1.0.0"
__author__  = "Hari Krishnan M R"
__project__ = "Deadlock Predictor"