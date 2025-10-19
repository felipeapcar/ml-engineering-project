"""
Slack notification service for fraud detection pipeline.
Sends alerts and status updates to Slack channels.
"""

import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import requests, but don't fail if not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available - Slack notifications will be logged only")


class SlackNotifier:
    """Send notifications to Slack via webhook."""

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL. If None, reads from SLACK_WEBHOOK_URL env var.
        """
        self.webhook_url = webhook_url or os.getenv('SLACK_WEBHOOK_URL')

        if not self.webhook_url:
            logger.warning(
                "No Slack webhook URL configured. "
                "Set SLACK_WEBHOOK_URL environment variable or pass webhook_url parameter."
            )

    def send_message(self, message: str, title: Optional[str] = None) -> bool:
        """
        Send a simple text message to Slack.

        Args:
            message: Message text
            title: Optional title/header

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.webhook_url:
            logger.info(f"[SLACK] {title or 'Message'}: {message}")
            return False

        if not REQUESTS_AVAILABLE:
            logger.error("requests library not installed - cannot send Slack message")
            return False

        payload = {
            "text": f"*{title}*\n{message}" if title else message
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Slack notification sent: {title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def send_training_success(self, metrics: Dict[str, float], run_id: str) -> bool:
        """
        Send notification for successful model training.

        Args:
            metrics: Model performance metrics
            run_id: MLflow run ID

        Returns:
            True if sent successfully
        """
        message = f"""
:white_check_mark: *Model Training Successful*

*Run ID:* `{run_id}`
*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Metrics:*
• ROC-AUC: `{metrics.get('roc_auc', 0):.4f}`
• Precision: `{metrics.get('precision', 0):.4f}`
• Recall: `{metrics.get('recall', 0):.4f}`
• F1-Score: `{metrics.get('f1_score', 0):.4f}`

The model is ready for validation and deployment.
        """.strip()

        return self.send_message(message, title="Fraud Detection Training")

    def send_deployment_success(
        self,
        metrics: Dict[str, float],
        improvement: Optional[float] = None
    ) -> bool:
        """
        Send notification for successful model deployment.

        Args:
            metrics: New model metrics
            improvement: ROC-AUC improvement over previous model

        Returns:
            True if sent successfully
        """
        improvement_text = ""
        if improvement is not None:
            improvement_text = f"\n*Improvement:* +{improvement:.2%} ROC-AUC"

        message = f"""
:rocket: *New Model Deployed to Production*

*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*New Model Metrics:*
• ROC-AUC: `{metrics.get('roc_auc', 0):.4f}`
• Precision: `{metrics.get('precision', 0):.4f}`
• Recall: `{metrics.get('recall', 0):.4f}`{improvement_text}

The new model is now serving predictions.
        """.strip()

        return self.send_message(message, title="Model Deployment")

    def send_validation_failure(
        self,
        validation_results: Dict,
        stage: str = "training"
    ) -> bool:
        """
        Send notification for validation failure.

        Args:
            validation_results: Validation check results
            stage: Stage where validation failed (training, deployment, etc.)

        Returns:
            True if sent successfully
        """
        failures = []
        for check_name, result in validation_results.items():
            if not result.get('valid', True):
                failures.append(f"• {check_name}: {result.get('message', 'Failed')}")

        failures_text = "\n".join(failures) if failures else "Unknown failure"

        message = f"""
:x: *Validation Failed - {stage.title()}*

*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Failed Checks:*
{failures_text}

Please review the logs and fix the issues before proceeding.
        """.strip()

        return self.send_message(message, title="Validation Failure")

    def send_no_deployment(self, reason: str, metrics: Dict[str, float]) -> bool:
        """
        Send notification when model is not deployed.

        Args:
            reason: Reason for not deploying
            metrics: New model metrics

        Returns:
            True if sent successfully
        """
        message = f"""
:warning: *Model Not Deployed*

*Reason:* {reason}

*New Model Metrics:*
• ROC-AUC: `{metrics.get('roc_auc', 0):.4f}`
• Precision: `{metrics.get('precision', 0):.4f}`
• Recall: `{metrics.get('recall', 0):.4f}`

Production model remains unchanged.
        """.strip()

        return self.send_message(message, title="Model Training Complete")

    def send_error_alert(self, error_message: str, task_id: str) -> bool:
        """
        Send alert for pipeline errors.

        Args:
            error_message: Error details
            task_id: Task that failed

        Returns:
            True if sent successfully
        """
        message = f"""
:rotating_light: *Pipeline Error Alert*

*Task:* `{task_id}`
*Timestamp:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Error:*
```
{error_message[:500]}
```

Please check the Airflow logs for full details.
        """.strip()

        return self.send_message(message, title="Pipeline Failure")


# Convenience function
def send_slack_notification(
    message: str,
    title: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a simple Slack notification.

    Args:
        message: Message text
        title: Optional title
        webhook_url: Optional webhook URL (uses env var if not provided)

    Returns:
        True if sent successfully
    """
    notifier = SlackNotifier(webhook_url)
    return notifier.send_message(message, title)
