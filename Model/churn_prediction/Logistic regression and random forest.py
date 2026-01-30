# churn_prediction/train_and_plot_models.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------
# Config: edit if needed
# -----------------------
DATA_PATH = r"C:\Users\OMEN\customer-lifetime-and-churn-analysis\Model\data\processed\customer_features.csv"
RESULTS_DIR = "results"
RANDOM_STATE = 42
CHURN_THRESHOLD_DAYS = 90  # used only if churn label missing

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------
# Utility plotting helpers
# -----------------------
def save_confusion_matrix(cm, out_png, title="Confusion Matrix", normalize=True):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_norm = cm

    plt.figure(figsize=(5, 5))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    classes = ["Active (0)", "Churn (1)"]
    plt.xticks([0, 1], classes, rotation=45)
    plt.yticks([0, 1], classes)
    plt.colorbar(label="Proportion" if normalize else "Count")

    # annotate with counts and (if normalized) percentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            text = f"{count}"
            if normalize:
                pct = cm_norm[i, j]
                text += f"\n{pct:.1%}"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_roc_curve(y_true, y_prob, out_png, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(linestyle=":", linewidth=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return auc


def save_precision_recall_threshold(y_true, y_prob, out_png, title="Precision vs Recall vs Threshold"):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds length = len(precisions)-1
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision", lw=2)
    plt.plot(thresholds, recalls[:-1], label="Recall", lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------
# Train + evaluate model
# -----------------------
def train_and_evaluate(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    results_txt_prefix,
    features=None,
):
    """Train model, evaluate, save text and plots. Returns (y_test, y_prob)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # some classifiers may not implement predict_proba (but these do)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: use decision_function normalized
        y_score = model.decision_function(X_test)
        y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    # metrics
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # save text
    txt_path = os.path.join(RESULTS_DIR, f"{results_txt_prefix}.txt")
    with open(txt_path, "w") as f:
        f.write(f"===== {model_name} =====\n\n")
        if features is not None:
            f.write("Features used: " + ", ".join(features) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix (raw counts):\n")
        f.write(str(cm) + "\n\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    print(f"[{model_name}] Results written to: {txt_path}")

    # save plots
    cm_png = os.path.join(RESULTS_DIR, f"{results_txt_prefix}_confusion.png")
    roc_png = os.path.join(RESULTS_DIR, f"{results_txt_prefix}_roc.png")
    pr_png = os.path.join(RESULTS_DIR, f"{results_txt_prefix}_pr_threshold.png")

    save_confusion_matrix(cm, cm_png, title=f"Confusion Matrix - {model_name}", normalize=True)
    save_roc_curve(y_test, y_prob, roc_png, title=f"ROC Curve - {model_name}")
    save_precision_recall_threshold(y_test, y_prob, pr_png, title=f"Precision/Recall vs Threshold - {model_name}")

    print(f"[{model_name}] Plots saved to: {cm_png}, {roc_png}, {pr_png}")

    # feature importance for RandomForest if requested
    if model_name.lower().startswith("random") and hasattr(model, "feature_importances_") and features is not None:
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fi_png = os.path.join(RESULTS_DIR, f"{results_txt_prefix}_feature_importance.png")
        plt.figure(figsize=(6, len(features) * 0.6))
        fi.plot(kind="barh")
        plt.title(f"Feature Importance - {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(fi_png, dpi=200)
        plt.close()
        # append importance to results txt
        with open(txt_path, "a") as f:
            f.write("\nFeature importances (ascending):\n")
            f.write(fi.to_string() + "\n")
        print(f"[{model_name}] Feature importance saved to: {fi_png}")

    return y_pred, y_prob


# -----------------------
# main pipeline
# -----------------------
def main():
    # load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Create churn label if missing (student project standard)
    if "churn" not in df.columns:
        df["churn"] = (df["recency"] > CHURN_THRESHOLD_DAYS).astype(int)
        print("Created churn label using recency >", CHURN_THRESHOLD_DAYS)

    # Features: NOTE we intentionally exclude 'recency' to avoid leakage
    features = ["frequency", "monetary", "avg_order_value", "tenure_days"]
    X = df[features]
    y = df["churn"]

    # Stratified random split (good when time column not available)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # -----------------------
    # Logistic Regression
    # -----------------------
    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    train_and_evaluate(
        lr_pipeline,
        model_name="Logistic Regression (no recency)",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        results_txt_prefix="logistic_regression_results",
        features=features,
    )

    # -----------------------
    # Random Forest
    # -----------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=15,
        min_samples_split=30,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    train_and_evaluate(
        rf,
        model_name="Random Forest (no recency)",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        results_txt_prefix="random_forest_results",
        features=features,
    )

    print("All done. Results and plots are in the 'results/' folder.")


if __name__ == "__main__":
    main()
