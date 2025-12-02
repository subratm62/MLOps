import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_recall_curve, auc
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-project")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/subratm62/tourism-project/Xtrain.csv"
Xtest_path = "hf://datasets/subratm62/tourism-project/Xtest.csv"
ytrain_path = "hf://datasets/subratm62/tourism-project/ytrain.csv"
ytest_path = "hf://datasets/subratm62/tourism-project/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# List of numerical features in the dataset
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "PitchSatisfactionScore", "NumberOfChildrenVisiting", "MonthlyIncome"
]

# List of categorical features in the dataset
categorical_nominal = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# List of binary features in the dataset
binary_features = ["Passport", "OwnCar"]

# ---------------------------------------------
# 3. Preprocessing Pipelines
# ---------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Keep binary columns as-is (just impute)
binary_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Combine all transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_nominal),
        ("bin", binary_transformer, binary_features)
    ]
)

# Set the clas weight to handle class imbalance
#class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

param_grid = {
    'gradientboostingclassifier__n_estimators': randint(100, 300),
    'gradientboostingclassifier__learning_rate': uniform(0.01, 0.2),
    'gradientboostingclassifier__max_depth': randint(2, 5),
    'gradientboostingclassifier__subsample': uniform(0.6, 0.4)
}

# Model pipeline
gb_model = GradientBoostingClassifier()
model_pipeline = make_pipeline(preprocessor, gb_model)

with mlflow.start_run():

    # Log class weight
    mlflow.log_param("scale_pos_weight", class_weight)

    # -------- Grid Search --------
    #grid_search = GridSearchCV(
    #    model_pipeline,
    #    param_grid,
    #    cv=5,
    #    n_jobs=-1,
    #    refit=True
    #)
    #grid_search.fit(Xtrain, ytrain)

    # -------- Random Search --------
    random_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        n_iter=50,           # Explore only 50 combinations
        scoring="recall",
        cv=5,
        n_jobs=5,
        random_state=42,
        verbose=1
    )

    ros = RandomOverSampler(random_state=42)
    Xtrain_resampled, ytrain_resampled = ros.fit_resample(Xtrain, ytrain)
    random_search.fit(Xtrain_resampled, ytrain_resampled)

    # Log every param set
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # ------------------------------
    #    Automatic Threshold Search
    # ------------------------------

    #y_scores = best_model.predict_proba(Xtest)[:, 1]
    #prec, rec, thr = precision_recall_curve(ytest, y_scores)

    #TARGET_RECALL = 0.90
    #idx = (rec >= TARGET_RECALL).argmax() if any(rec >= TARGET_RECALL) else rec.argmax()
    #best_threshold = thr[idx]

    classification_threshold = 0.40
    #classification_threshold = best_threshold

    mlflow.log_metric("best_threshold", float(best_threshold))
    mlflow.log_metric("best_threshold_precision", float(precisions[best_idx]))
    mlflow.log_metric("best_threshold_recall", float(recalls[best_idx]))
    mlflow.log_metric("best_threshold_f1", float(f1_scores[best_idx]))

    # ------------------------------
    #  Predictions using best threshold
    # ------------------------------

    # Train predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= best_threshold).astype(int)

    # Test predictions
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= best_threshold).astype(int)

    # Classification reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log key metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score']
    })

    # Persist threshold with the model artifacts
    #with open("chosen_threshold.txt", "w") as f:
    #    f.write(str(best_threshold))

    #mlflow.log_artifact("chosen_threshold.txt")

    meta = {
        "threshold": float(classification_threshold),
        "strategy": "auto",
        "test_precision": float(test_report['1']['precision']),
        "test_recall": float(test_report['1']['recall']),
        "test_f1": float(test_report['1']['f1-score'])
    }

    # Save the model locally
    model_path = "best_tourism_model.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, name="model")
    print(f"Model saved as artifact at: {model_path}")

    # Log model meta
    with open("model_meta.json", "w") as f:
        json.dump(meta, f)
    mlflow.log_artifact("model_meta.json")

    # Upload to Hugging Face
    repo_id = "subratm62/tourism-project"
    repo_type = "model"
    
    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_model.joblib",
        path_in_repo="best_tourism_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

    # Save the full pipeline (preprocessor + model)
    mlflow.sklearn.log_model(best_model, "model")
