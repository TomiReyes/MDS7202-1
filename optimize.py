import mlflow
import optuna
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import mlflow.models.signature
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model

def optimize_model(df, target_column='Potability', n_trials=50):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    experiment_name = f"XGBoost_Optimization_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    mlflow.set_experiment(experiment_name)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),  
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),  
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20), 
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),  
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),  
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10), 
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10), 
        }

        formatted_lr = f"{params['learning_rate']:.2f}"
        run_name = f"XGBoost_lr_{formatted_lr}_depth_{params['max_depth']}"

        with mlflow.start_run(run_name=run_name):
            model = XGBClassifier(**params, eval_metric="logloss")
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred)

            input_example = pd.DataFrame(X_train_scaled[:5], columns=X.columns)
            signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))

            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)

            mlflow.sklearn.log_model(
                model, artifact_path="model", input_example=input_example, signature=signature
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(X.columns, model.feature_importances_)
            ax.set_title("Feature Importance")
            ax.set_ylabel("Importance")
            plt.tight_layout()
            plot_path = f"./plots/feature_importance_trial_{trial.number}.png"
            os.makedirs("./plots", exist_ok=True)
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="plots")
            plt.close(fig)

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    best_model = get_best_model(experiment_id)

    model_path = "./models/best_model.pkl"
    os.makedirs("./models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    mlflow.log_artifact(model_path, artifact_path="models")

    mlflow.log_artifact("requirements.txt", artifact_path="dependencies")

    optuna.visualization.plot_param_importances(study).write_html("./plots/param_importance.html")
    mlflow.log_artifact("./plots/param_importance.html", artifact_path="plots")

    optuna.visualization.plot_optimization_history(study).write_html("./plots/optimization_history.html")
    mlflow.log_artifact("./plots/optimization_history.html", artifact_path="plots")

    print(f"Optimization completed. Best params: {study.best_params}")
    print(f"Experiment ID: {experiment_id}")

if __name__ == "__main__":
    df = pd.read_csv("water_potability.csv")  
    optimize_model(df, n_trials=100)
