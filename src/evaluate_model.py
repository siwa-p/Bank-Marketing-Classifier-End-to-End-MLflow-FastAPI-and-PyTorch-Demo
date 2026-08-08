import os

import duckdb
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utilities.logging_config import logger
from utilities.utils import process_features

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuckLakeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        features_tot = process_features(data)
        self.X = torch.tensor(features_tot.astype('float32').values, dtype=torch.float32)
        self.y = torch.tensor(data['y'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

con = duckdb.connect(database=':memory:')
folder_path = os.getenv("DUCKLAKE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ducklake_db_path = os.path.join(folder_path, "catalog.ducklake")
ducklake_data_path = os.path.join(folder_path, "catalog_data")
con.execute(f"ATTACH 'ducklake:{ducklake_db_path}' AS my_lake (DATA_PATH '{ducklake_data_path}')")
con.execute("USE my_lake")

# Same split as bank_classifier.py (random_state=1) so this is the held-out
# validation slice the registered model was scored on during training.
train_data = con.execute("SELECT * FROM bank_schema.train").fetchdf()
_, val_data = train_test_split(train_data, test_size=0.2, random_state=1)
logger.info(f"Validation data shape: {val_data.shape}")

val_dataset = DuckLakeDataset(val_data)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

client = mlflow.MlflowClient()
model_version = client.get_model_version_by_alias("bank_classifier", "production")
run_id = model_version.run_id
logger.info(f"Evaluating production model (version {model_version.version}, run {run_id})")

model = mlflow.pytorch.load_model(f"models:/bank_classifier@production", map_location=device)
model.eval()
model.to(device)

all_probs, all_targets = [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).squeeze(1)
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())

y_true = np.array(all_targets)
y_prob = np.array(all_probs)
y_pred = (y_prob > 0.5).astype(int)

roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
f1_at_50 = f1_score(y_true, y_pred)

thresholds = np.linspace(0.05, 0.95, 19)
f1_by_threshold = [f1_score(y_true, (y_prob > t).astype(int)) for t in thresholds]
best_idx = int(np.argmax(f1_by_threshold))
best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_by_threshold[best_idx])

logger.info(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, F1@0.5: {f1_at_50:.4f}")
logger.info(f"Best F1 {best_f1:.4f} at threshold {best_threshold:.2f}")

fig_roc, ax = plt.subplots(figsize=(5, 5))
fpr, tpr, _ = roc_curve(y_true, y_prob)
ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (validation)")
ax.legend()
fig_roc.tight_layout()

fig_pr, ax = plt.subplots(figsize=(5, 5))
precision, recall, _ = precision_recall_curve(y_true, y_prob)
base_rate = y_true.mean()
ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
ax.axhline(base_rate, linestyle="--", color="gray", label=f"base rate = {base_rate:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (validation)")
ax.legend()
fig_pr.tight_layout()

fig_cm, ax = plt.subplots(figsize=(5, 5))
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["no", "yes"]).plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix @ threshold 0.5 (validation)")
fig_cm.tight_layout()

fig_cal, ax = plt.subplots(figsize=(5, 5))
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
ax.plot(prob_pred, prob_true, marker="o", label="model")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve (validation)")
ax.legend()
fig_cal.tight_layout()

fig_f1, ax = plt.subplots(figsize=(5, 5))
ax.plot(thresholds, f1_by_threshold, marker="o")
ax.axvline(best_threshold, linestyle="--", color="gray", label=f"best = {best_threshold:.2f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("F1 score")
ax.set_title("F1 vs. Threshold (validation)")
ax.legend()
fig_f1.tight_layout()

note = f"""## Post-hoc evaluation — production model v{model_version.version}

Computed on the same validation split used during training (`train_test_split`,
`test_size=0.2`, `random_state=1`), {len(y_true):,} rows, base rate {base_rate:.1%} positive.

| Metric | Value |
|---|---|
| ROC-AUC | {roc_auc:.4f} |
| PR-AUC (avg. precision) | {pr_auc:.4f} |
| F1 @ threshold 0.5 | {f1_at_50:.4f} |
| Best F1 | {best_f1:.4f} (at threshold {best_threshold:.2f}) |

At the default 0.5 threshold the model favors precision over recall on the
minority class — see `f1_vs_threshold.png` for the tradeoff and
`confusion_matrix.png` for the breakdown. `calibration_curve.png` shows how
well predicted probabilities track observed frequencies, relevant since
`predictions.csv` ships raw probabilities rather than thresholded labels.
"""

with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics({
        "eval_roc_auc": roc_auc,
        "eval_pr_auc": pr_auc,
        "eval_f1_at_0.5": f1_at_50,
        "eval_best_f1": best_f1,
        "eval_best_f1_threshold": best_threshold,
    })
    mlflow.log_figure(fig_roc, "evaluation/roc_curve.png")
    mlflow.log_figure(fig_pr, "evaluation/precision_recall_curve.png")
    mlflow.log_figure(fig_cm, "evaluation/confusion_matrix.png")
    mlflow.log_figure(fig_cal, "evaluation/calibration_curve.png")
    mlflow.log_figure(fig_f1, "evaluation/f1_vs_threshold.png")
    mlflow.set_tag("mlflow.note.content", note)

logger.info(f"Logged evaluation graphs and notes to run {run_id}")
print(f"Logged evaluation graphs and notes to run {run_id} — view at {mlflow.get_tracking_uri()}/#/experiments/0/runs/{run_id}")
