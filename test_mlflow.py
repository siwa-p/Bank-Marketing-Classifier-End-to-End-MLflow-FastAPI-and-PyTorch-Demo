import mlflow
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("synthetic_regression_test")

X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

print("X shape:", X.shape)

# Log to MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_params({"n_samples": 100, "n_features": 5})
    lr = LinearRegression()
    model = lr.fit(X, y)  # Train the model
    y_pred = model.predict(X)  # Predict
    mse = mean_squared_error(y, y_pred)  # Evaluate
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model", registered_model_name="LinearRegressionModel", input_example=X[:5].tolist())