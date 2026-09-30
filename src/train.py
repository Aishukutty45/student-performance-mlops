import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/student-mat.csv", sep=";")

print("========== MODEL TRAINING ==========")

print("\nDataset loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================
# 2. REMOVE G1 AND G2
# ==========================================

# We remove previous grades to make this an
# early student performance prediction system.

df = df.drop(columns=["G1", "G2"])

print("\nG1 and G2 removed")
print("Remaining columns:", len(df.columns))


# ==========================================
# 3. TARGET AND FEATURES
# ==========================================

X = df.drop(columns=["G3"])
y = df["G3"]

print("\nTarget Variable: G3")
print("Number of Features:", X.shape[1])


# ==========================================
# 4. IDENTIFY DATA TYPES
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:", len(categorical_columns))
print("Numerical Features:", len(numerical_columns))


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 7. DEFINE ML MODELS
# ==========================================

models = {
    "LinearRegression": LinearRegression(),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "GradientBoosting": GradientBoostingRegressor(
        random_state=42
    )
}


# ==========================================
# 8. MLflow EXPERIMENT
# ==========================================

mlflow.set_experiment(
    "Student_Performance_Prediction"
)


# ==========================================
# 9. TRAIN AND COMPARE MODELS
# ==========================================

best_model = None
best_rmse = float("inf")
best_name = None

results = []


for name, model in models.items():

    print("\n----------------------------------")
    print("Training:", name)

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)

        # MLflow logging
        mlflow.log_param(
            "model",
            name
        )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "RMSE",
            rmse
        )

        mlflow.log_metric(
            "R2",
            r2
        )

        # Store result
        results.append(
            [name, mae, rmse, r2]
        )

        # Select best model based on RMSE
        if rmse < best_rmse:

            best_rmse = rmse
            best_model = pipeline
            best_name = name


# ==========================================
# 10. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/student_model.pkl"
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n")
print("==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    f"{'Model':<20}"
    f"{'MAE':<15}"
    f"{'RMSE':<15}"
    f"{'R2':<15}"
)

for result in results:

    print(
        f"{result[0]:<20}"
        f"{result[1]:<15.4f}"
        f"{result[2]:<15.4f}"
        f"{result[3]:<15.4f}"
    )


print("\n==========================================")
print("BEST MODEL:", best_name)
print("BEST RMSE :", best_rmse)
print("==========================================")

print("\nBest model saved to:")
print("models/student_model.pkl")

print("\n========== TRAINING COMPLETED ==========")