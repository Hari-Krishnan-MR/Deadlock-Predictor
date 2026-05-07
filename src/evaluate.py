# evaluate.py
# Project: Deadlock Predictor - Deadlock-1.0
# Purpose: Model evaluation, ablation, cross-validation
# Author: Hari Krishnan M R
# Date: May 2026

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from scipy import stats
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(name, y_true, y_pred, y_prob):
    """
    Calculate and print all evaluation metrics.

    Parameters
    ----------
    name : str
        Model name for display
    y_true : array
        Actual labels
    y_pred : array
        Predicted labels
    y_prob : array
        Predicted probabilities

    Returns
    -------
    dict
        All metric values
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal(0)', 'Deadlock(1)']
    ))

    return {
        'model'    : name,
        'accuracy' : round(acc, 4),
        'precision': round(prec, 4),
        'recall'   : round(rec, 4),
        'f1'       : round(f1, 4),
        'roc_auc'  : round(auc, 4)
    }


def run_ablation_study(X_train, X_test,
                        y_train, y_test,
                        novel_features,
                        all_features,
                        scale_pos_weight,
                        baseline_f1,
                        baseline_auc,
                        random_state=42):
    """
    Ablation study - remove each novel feature one
    at a time and measure F1 and AUC drop.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    novel_features : list
        List of novel feature names
    all_features : list
        Full feature list
    scale_pos_weight : float
        Class imbalance ratio
    baseline_f1 : float
        F1 of full model
    baseline_auc : float
        AUC of full model
    random_state : int

    Returns
    -------
    pd.DataFrame
        Ablation results ranked by F1 impact
    """
    print(f"Baseline F1  : {baseline_f1:.4f}")
    print(f"Baseline AUC : {baseline_auc:.4f}")
    print("-" * 55)

    ablation_results = []

    for feature in novel_features:
        features_without = [
            f for f in all_features
            if f != feature
        ]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(
            X_train[features_without], y_train
        )

        y_pred = model.predict(
            X_test[features_without]
        )
        y_prob = model.predict_proba(
            X_test[features_without]
        )[:, 1]

        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1_drop  = baseline_f1 - f1
        auc_drop = baseline_auc - auc

        ablation_results.append({
            'Removed Feature' : feature,
            'F1'              : round(f1, 4),
            'F1 Drop'         : round(f1_drop, 4),
            'AUC'             : round(auc, 4),
            'AUC Drop'        : round(auc_drop, 4)
        })

        print(f"Without {feature:<30} "
              f"F1={f1:.4f} "
              f"(drop={f1_drop:+.4f})")

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df = ablation_df.sort_values(
        'F1 Drop', ascending=False
    )
    return ablation_df


def run_cross_validation(model, X, y,
                          n_splits=5,
                          random_state=42):
    """
    Run stratified k-fold cross validation.

    Parameters
    ----------
    model : unfitted model
    X : pd.DataFrame
    y : pd.Series
    n_splits : int
        Number of folds (default 5)
    random_state : int

    Returns
    -------
    dict
        Mean and std for each metric
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=[
            'f1', 'roc_auc',
            'precision', 'recall'
        ],
        return_train_score=False
    )

    results = {}
    for metric in ['f1', 'roc_auc',
                   'precision', 'recall']:
        key = f'test_{metric}'
        mean = scores[key].mean()
        std  = scores[key].std()
        results[metric] = {
            'mean': round(mean, 4),
            'std' : round(std, 4)
        }
        print(f"{metric:<12}: "
              f"{mean:.4f} +/- {std:.4f}")

    return results


def run_paired_ttest(cv_scores_best,
                     cv_scores_baseline):
    """
    Paired t-test between best model and baseline.

    Parameters
    ----------
    cv_scores_best : array
        Per-fold F1 scores of best model
    cv_scores_baseline : array
        Per-fold F1 scores of baseline model

    Returns
    -------
    tuple
        (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(
        cv_scores_best,
        cv_scores_baseline
    )

    print(f"t-statistic : {t_stat:.4f}")
    print(f"p-value     : {p_value:.6f}")

    if p_value < 0.05:
        print("Result      : SIGNIFICANT (p < 0.05)")
    else:
        print("Result      : Not significant")

    return t_stat, p_value


def plot_roc_curves(models_roc, save_path=None):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    models_roc : list of tuples
        [(name, y_true, y_prob, color), ...]
    save_path : str or None
        Path to save figure
    """
    plt.figure(figsize=(10, 7))

    for name, y_true, y_prob, color in models_roc:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f'{name} (AUC={auc:.4f})'
        )

    plt.plot(
        [0, 1], [0, 1],
        'k--', linewidth=1,
        label='Random (AUC=0.5)'
    )
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(
        'ROC Curves — All Models',
        fontsize=13, fontweight='bold'
    )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"ROC curves saved: {save_path}")
    plt.show()


def plot_confusion_matrices(cms, save_path=None):
    """
    Plot confusion matrices for multiple models.

    Parameters
    ----------
    cms : list of tuples
        [(name, y_true, y_pred), ...]
    save_path : str or None
    """
    n = len(cms)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(18, rows * 5)
    )
    axes = axes.flatten()
    fig.suptitle(
        'Confusion Matrices — All Models',
        fontsize=13, fontweight='bold'
    )

    for i, (name, y_true, y_pred) in enumerate(cms):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt=',',
            cmap='Blues',
            ax=axes[i],
            xticklabels=['Normal', 'Deadlock'],
            yticklabels=['Normal', 'Deadlock']
        )
        axes[i].set_title(
            name, fontsize=10, fontweight='bold'
        )
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrices saved: {save_path}")
    plt.show()


def plot_ablation(ablation_df, save_path=None):
    """
    Plot ablation study results.

    Parameters
    ----------
    ablation_df : pd.DataFrame
        Output from run_ablation_study()
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Ablation Study — Novel Feature Contribution',
        fontweight='bold'
    )

    colors = [
        'crimson' if x > 0.01 else 'orange'
        for x in ablation_df['F1 Drop']
    ]

    axes[0].barh(
        ablation_df['Removed Feature'],
        ablation_df['F1 Drop'],
        color=colors, edgecolor='black'
    )
    axes[0].set_title('F1 Drop When Feature Removed')
    axes[0].set_xlabel('F1 Drop')

    axes[1].barh(
        ablation_df['Removed Feature'],
        ablation_df['AUC Drop'],
        color=colors, edgecolor='black'
    )
    axes[1].set_title('AUC Drop When Feature Removed')
    axes[1].set_xlabel('AUC Drop')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Ablation plot saved: {save_path}")
    plt.show()