"""
Email notification service for fraud detection pipeline.
Sends alerts and status reports via email.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email notifications via SMTP."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None
    ):
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server host (default: from SMTP_HOST env var)
            smtp_port: SMTP server port (default: from SMTP_PORT env var or 587)
            smtp_user: SMTP username (default: from SMTP_USER env var)
            smtp_password: SMTP password (default: from SMTP_PASSWORD env var)
            from_email: From email address (default: from FROM_EMAIL env var)
        """
        self.smtp_host = smtp_host or os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = smtp_user or os.getenv('SMTP_USER')
        self.smtp_password = smtp_password or os.getenv('SMTP_PASSWORD')
        self.from_email = from_email or os.getenv('FROM_EMAIL') or self.smtp_user

        if not all([self.smtp_user, self.smtp_password]):
            logger.warning(
                "Email credentials not configured. "
                "Set SMTP_USER and SMTP_PASSWORD environment variables."
            )

    def send_email(
        self,
        to_emails: List[str],
        subject: str,
        body: str,
        html: bool = False
    ) -> bool:
        """
        Send an email.

        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            body: Email body content
            html: If True, body is HTML; otherwise plain text

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.smtp_user or not self.smtp_password:
            logger.info(f"[EMAIL] To: {to_emails}, Subject: {subject}")
            logger.info(f"[EMAIL] Body: {body[:200]}...")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject

            # Attach body
            mime_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, mime_type))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {to_emails}: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_training_report(
        self,
        to_emails: List[str],
        metrics: Dict[str, float],
        run_id: str,
        success: bool = True
    ) -> bool:
        """
        Send training completion report.

        Args:
            to_emails: Recipient email addresses
            metrics: Model performance metrics
            run_id: MLflow run ID
            success: Whether training was successful

        Returns:
            True if sent successfully
        """
        status = "Successful" if success else "Failed"
        subject = f"Fraud Detection Model Training {status}"

        body = f"""
Model Training Report
{'=' * 50}

Status: {status}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Run ID: {run_id}

Performance Metrics:
-------------------
ROC-AUC:    {metrics.get('roc_auc', 0):.4f}
Precision:  {metrics.get('precision', 0):.4f}
Recall:     {metrics.get('recall', 0):.4f}
F1-Score:   {metrics.get('f1_score', 0):.4f}

{'The model is ready for validation and deployment.' if success else 'Please review the logs for error details.'}

---
This is an automated message from the Fraud Detection Pipeline.
        """.strip()

        return self.send_email(to_emails, subject, body)

    def send_deployment_notification(
        self,
        to_emails: List[str],
        metrics: Dict[str, float],
        deployed: bool = True,
        reason: Optional[str] = None
    ) -> bool:
        """
        Send deployment notification.

        Args:
            to_emails: Recipient email addresses
            metrics: Model metrics
            deployed: Whether model was deployed
            reason: Reason for deployment decision

        Returns:
            True if sent successfully
        """
        if deployed:
            subject = "New Fraud Detection Model Deployed"
            status_text = "A new model has been deployed to production."
        else:
            subject = "Fraud Detection Model Training Complete - Not Deployed"
            status_text = f"Model was not deployed. Reason: {reason or 'Unknown'}"

        body = f"""
Model Deployment Notification
{'=' * 50}

{status_text}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Metrics:
--------------
ROC-AUC:    {metrics.get('roc_auc', 0):.4f}
Precision:  {metrics.get('precision', 0):.4f}
Recall:     {metrics.get('recall', 0):.4f}
F1-Score:   {metrics.get('f1_score', 0):.4f}

{'Production API is now using the new model.' if deployed else 'Production model remains unchanged.'}

---
This is an automated message from the Fraud Detection Pipeline.
        """.strip()

        return self.send_email(to_emails, subject, body)

    def send_validation_failure_report(
        self,
        to_emails: List[str],
        validation_results: Dict,
        stage: str = "training"
    ) -> bool:
        """
        Send validation failure report.

        Args:
            to_emails: Recipient email addresses
            validation_results: Validation check results
            stage: Stage where validation failed

        Returns:
            True if sent successfully
        """
        subject = f"Validation Failed - {stage.title()} Stage"

        failures = []
        for check_name, result in validation_results.items():
            if not result.get('valid', True):
                failures.append(f"  - {check_name}: {result.get('message', 'Failed')}")

        failures_text = "\n".join(failures) if failures else "  No specific failures recorded"

        body = f"""
Validation Failure Report
{'=' * 50}

Stage: {stage.title()}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Failed Validation Checks:
-------------------------
{failures_text}

Action Required:
----------------
Please review the validation results and address the issues
before re-running the pipeline.

Check the Airflow logs for detailed error messages.

---
This is an automated message from the Fraud Detection Pipeline.
        """.strip()

        return self.send_email(to_emails, subject, body)

    def send_error_alert(
        self,
        to_emails: List[str],
        error_message: str,
        task_id: str,
        dag_id: str
    ) -> bool:
        """
        Send error alert email.

        Args:
            to_emails: Recipient email addresses
            error_message: Error details
            task_id: Task that failed
            dag_id: DAG ID

        Returns:
            True if sent successfully
        """
        subject = f"Pipeline Error Alert - {dag_id}"

        body = f"""
Pipeline Error Alert
{'=' * 50}

DAG: {dag_id}
Task: {task_id}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Error Details:
--------------
{error_message[:1000]}

Action Required:
----------------
The pipeline has failed and requires attention.
Please check the Airflow UI for full logs and stack traces.

Airflow UI: http://localhost:8080

---
This is an automated message from the Fraud Detection Pipeline.
        """.strip()

        return self.send_email(to_emails, subject, body)


# Convenience function
def send_email_notification(
    to_emails: List[str],
    subject: str,
    body: str,
    **kwargs
) -> bool:
    """
    Send a simple email notification.

    Args:
        to_emails: Recipient email addresses
        subject: Email subject
        body: Email body
        **kwargs: Additional EmailNotifier init parameters

    Returns:
        True if sent successfully
    """
    notifier = EmailNotifier(**kwargs)
    return notifier.send_email(to_emails, subject, body)
