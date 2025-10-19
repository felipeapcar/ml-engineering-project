"""
Model validation checks for fraud detection pipeline.
Ensures model performance meets production standards.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates model performance and quality."""

    # Minimum performance thresholds
    MIN_ROC_AUC = 0.85
    MIN_PRECISION = 0.70
    MIN_RECALL = 0.60
    MIN_F1_SCORE = 0.65

    # Model comparison thresholds
    MIN_IMPROVEMENT = 0.01  # 1% improvement in ROC-AUC
    MAX_RECALL_DROP = 0.02  # Max 2% drop in recall allowed

    def __init__(self, model_path: str, metrics: Dict[str, float]):
        self.model_path = Path(model_path)
        self.metrics = metrics
        self.validation_results = {}

    def validate_all(self) -> Tuple[bool, Dict]:
        """
        Run all model validation checks.

        Returns:
            Tuple of (is_valid, validation_results)
        """
        checks = [
            self.check_model_exists,
            self.check_model_loadable,
            self.check_performance_thresholds,
            self.check_metrics_sanity
        ]

        all_valid = True
        for check in checks:
            try:
                is_valid, message = check()
                self.validation_results[check.__name__] = {
                    'valid': is_valid,
                    'message': message
                }
                if not is_valid:
                    all_valid = False
                    logger.warning(f"{check.__name__} failed: {message}")
            except Exception as e:
                self.validation_results[check.__name__] = {
                    'valid': False,
                    'message': f"Exception: {str(e)}"
                }
                all_valid = False
                logger.error(f"{check.__name__} raised exception: {e}")

        return all_valid, self.validation_results

    def check_model_exists(self) -> Tuple[bool, str]:
        """Check if model file exists."""
        if not self.model_path.exists():
            return False, f"Model file not found: {self.model_path}"
        return True, f"Model file exists: {self.model_path.name}"

    def check_model_loadable(self) -> Tuple[bool, str]:
        """Check if model can be loaded successfully."""
        try:
            model = joblib.load(self.model_path)

            # Check if model has predict method
            if not hasattr(model, 'predict'):
                return False, "Model missing 'predict' method"

            # Check if model has predict_proba method (needed for fraud detection)
            if not hasattr(model, 'predict_proba'):
                return False, "Model missing 'predict_proba' method"

            return True, f"Model loaded successfully: {type(model).__name__}"

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def check_performance_thresholds(self) -> Tuple[bool, str]:
        """Validate that model meets minimum performance thresholds."""
        failures = []

        if self.metrics.get('roc_auc', 0) < self.MIN_ROC_AUC:
            failures.append(
                f"ROC-AUC {self.metrics['roc_auc']:.4f} < {self.MIN_ROC_AUC}"
            )

        if self.metrics.get('precision', 0) < self.MIN_PRECISION:
            failures.append(
                f"Precision {self.metrics['precision']:.4f} < {self.MIN_PRECISION}"
            )

        if self.metrics.get('recall', 0) < self.MIN_RECALL:
            failures.append(
                f"Recall {self.metrics['recall']:.4f} < {self.MIN_RECALL}"
            )

        if self.metrics.get('f1_score', 0) < self.MIN_F1_SCORE:
            failures.append(
                f"F1-Score {self.metrics['f1_score']:.4f} < {self.MIN_F1_SCORE}"
            )

        if failures:
            return False, "Performance below thresholds: " + "; ".join(failures)

        return True, (
            f"All metrics above thresholds: "
            f"ROC-AUC={self.metrics['roc_auc']:.4f}, "
            f"Precision={self.metrics['precision']:.4f}, "
            f"Recall={self.metrics['recall']:.4f}"
        )

    def check_metrics_sanity(self) -> Tuple[bool, str]:
        """Sanity check on metrics values."""
        issues = []

        # Check all metrics are between 0 and 1
        for metric_name, value in self.metrics.items():
            if not (0 <= value <= 1):
                issues.append(f"{metric_name}={value} not in [0, 1]")

        # Check precision/recall trade-off is reasonable
        precision = self.metrics.get('precision', 0)
        recall = self.metrics.get('recall', 0)

        if precision > 0.99 and recall < 0.3:
            issues.append("Precision too high, recall too low - model may be too conservative")

        if recall > 0.99 and precision < 0.3:
            issues.append("Recall too high, precision too low - model may be too aggressive")

        if issues:
            return False, "Metric sanity checks failed: " + "; ".join(issues)

        return True, "Metrics are within reasonable ranges"

    def compare_with_production(
        self, production_metrics_path: str
    ) -> Tuple[bool, str]:
        """
        Compare new model with current production model.

        Args:
            production_metrics_path: Path to production model metrics JSON

        Returns:
            Tuple of (should_deploy, reason)
        """
        prod_metrics_file = Path(production_metrics_path)

        # If no production model exists, deploy new model
        if not prod_metrics_file.exists():
            return True, "No production model exists - deploying new model"

        # Load production metrics
        try:
            with open(prod_metrics_file, 'r') as f:
                prod_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load production metrics: {e}")
            return False, f"Cannot compare with production: {str(e)}"

        # Compare ROC-AUC
        new_roc_auc = self.metrics.get('roc_auc', 0)
        prod_roc_auc = prod_metrics.get('roc_auc', 0)

        roc_auc_improvement = new_roc_auc - prod_roc_auc

        # Compare Precision
        new_precision = self.metrics.get('precision', 0)
        prod_precision = prod_metrics.get('precision', 0)

        # Compare Recall
        new_recall = self.metrics.get('recall', 0)
        prod_recall = prod_metrics.get('recall', 0)

        recall_drop = prod_recall - new_recall

        # Decision logic
        if roc_auc_improvement < self.MIN_IMPROVEMENT:
            return False, (
                f"ROC-AUC improvement {roc_auc_improvement:.4f} "
                f"< threshold {self.MIN_IMPROVEMENT}"
            )

        if new_precision < prod_precision:
            logger.warning(
                f"Precision decreased: {prod_precision:.4f} -> {new_precision:.4f}"
            )

        if recall_drop > self.MAX_RECALL_DROP:
            return False, (
                f"Recall dropped too much: {prod_recall:.4f} -> {new_recall:.4f} "
                f"(drop: {recall_drop:.4f} > {self.MAX_RECALL_DROP})"
            )

        # Model is better
        return True, (
            f"Model improved: "
            f"ROC-AUC {prod_roc_auc:.4f} -> {new_roc_auc:.4f} "
            f"(+{roc_auc_improvement:.4f})"
        )


def validate_model(
    model_path: str,
    metrics: Dict[str, float],
    production_metrics_path: Optional[str] = None
) -> Tuple[bool, Dict]:
    """
    Convenience function to validate model.

    Args:
        model_path: Path to model file
        metrics: Dictionary of model metrics
        production_metrics_path: Optional path to production metrics for comparison

    Returns:
        Tuple of (is_valid, validation_results)
    """
    validator = ModelValidator(model_path, metrics)
    is_valid, results = validator.validate_all()

    # Add comparison result if production metrics provided
    if production_metrics_path:
        should_deploy, reason = validator.compare_with_production(production_metrics_path)
        results['production_comparison'] = {
            'valid': should_deploy,
            'message': reason
        }
        if not should_deploy:
            is_valid = False

    return is_valid, results
