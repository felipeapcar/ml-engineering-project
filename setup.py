from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "mlflow",
        # agrega cualquier otra dependencia que uses
    ],
    entry_points={
        "console_scripts": [
            "run-fraud-pipeline=scripts.run_pipeline:main",
        ],
    },
)