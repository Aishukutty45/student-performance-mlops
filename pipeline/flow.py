from prefect import flow, task
import subprocess
import sys


@task
def validate_data():
    print("\n========== DATA VALIDATION ==========")

    subprocess.run(
        [sys.executable, "src/check_data.py"],
        check=True
    )

    print("\nData validation completed successfully.")


@task
def train_model():
    print("\n========== MODEL TRAINING ==========")

    subprocess.run(
        [sys.executable, "src/train.py"],
        check=True
    )

    print("\nModel training completed successfully.")


@flow(name="student-performance-pipeline")
def student_performance_pipeline():

    print("\n========================================")
    print("STUDENT PERFORMANCE MLOPS PIPELINE")
    print("========================================")

    validate_data()

    train_model()

    print("\n========================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    student_performance_pipeline()