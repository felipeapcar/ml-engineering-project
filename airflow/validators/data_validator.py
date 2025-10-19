"""
Data validation checks for fraud detection pipeline.
Ensures data quality before training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and schema for fraud detection."""

    REQUIRED_COLUMNS = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.validation_results = {}

    def validate_all(self) -> Tuple[bool, Dict]:
        """
        Run all validation checks.

        Returns:
            Tuple of (is_valid, validation_results)
        """
        checks = [
            self.check_file_exists,
            self.check_schema,
            self.check_data_quality,
            self.check_class_distribution,
            self.check_feature_ranges
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

    def check_file_exists(self) -> Tuple[bool, str]:
        """Check if data file exists."""
        if not self.data_path.exists():
            return False, f"Data file not found: {self.data_path}"
        return True, "Data file exists"

    def check_schema(self) -> Tuple[bool, str]:
        """Validate that all required columns are present."""
        df = pd.read_csv(self.data_path, nrows=10)

        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        extra_cols = set(df.columns) - set(self.REQUIRED_COLUMNS)

        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")

        return True, f"Schema valid: {len(df.columns)} columns"

    def check_data_quality(self) -> Tuple[bool, str]:
        """Check for missing values and data types."""
        df = pd.read_csv(self.data_path)

        # Check missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            missing_pct = (missing_count / (df.shape[0] * df.shape[1])) * 100
            if missing_pct > 5:  # Threshold: 5%
                return False, f"Too many missing values: {missing_pct:.2f}%"
            logger.warning(f"Found {missing_count} missing values ({missing_pct:.2f}%)")

        # Check data types
        if df['Class'].dtype not in ['int64', 'int32', 'float64']:
            return False, f"Invalid Class column type: {df['Class'].dtype}"

        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            logger.warning(f"Found {duplicates} duplicate rows ({dup_pct:.2f}%)")

        return True, f"Data quality OK: {len(df)} rows, {missing_count} missing values"

    def check_class_distribution(self) -> Tuple[bool, str]:
        """Validate class distribution for fraud detection."""
        df = pd.read_csv(self.data_path)

        class_counts = df['Class'].value_counts()

        if len(class_counts) != 2:
            return False, f"Expected 2 classes, found {len(class_counts)}"

        fraud_pct = (class_counts.get(1, 0) / len(df)) * 100

        # Typical fraud rate is 0.1% - 2%
        if fraud_pct < 0.01:
            return False, f"Fraud rate too low: {fraud_pct:.4f}%"
        if fraud_pct > 10:
            return False, f"Fraud rate suspiciously high: {fraud_pct:.2f}%"

        return True, f"Class distribution OK: {fraud_pct:.4f}% fraud"

    def check_feature_ranges(self) -> Tuple[bool, str]:
        """Check if feature values are within expected ranges."""
        df = pd.read_csv(self.data_path)

        # Check Amount
        if df['Amount'].min() < 0:
            return False, "Negative amounts found"

        max_amount = df['Amount'].max()
        if max_amount > 100000:  # Suspiciously high
            logger.warning(f"Very high transaction amount detected: {max_amount}")

        # Check Time
        if df['Time'].min() < 0:
            return False, "Negative time values found"

        # Check PCA components (V1-V28) - should be roughly normalized
        v_columns = [f'V{i}' for i in range(1, 29)]
        v_means = df[v_columns].mean()
        v_stds = df[v_columns].std()

        # PCA components should be roughly mean=0, std=1
        if (abs(v_means) > 2).any():
            logger.warning("PCA components not centered")

        return True, f"Feature ranges OK: Amount max={max_amount:.2f}"


def validate_data(data_path: str) -> Tuple[bool, Dict]:
    """
    Convenience function to validate data.

    Args:
        data_path: Path to CSV file

    Returns:
        Tuple of (is_valid, validation_results)
    """
    validator = DataValidator(data_path)
    return validator.validate_all()
