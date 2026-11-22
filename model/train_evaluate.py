import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. Load preprocessed split & preprocessor fitted strictly on training data
artifact_path = "model/artifacts/preprocessed_split.pkl"
if not os.path.exists(artifact_path):
    raise FileNotFoundError("Run 'python model/preprocess.py' first to generate preprocessed_split.pkl")

data = joblib.load(artifact_path)
preprocessor = data["preprocessor"]
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# 2. Define the 5 Required Classification Models
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=RANDOM_SEED),
    "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski"),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED)
}

os.makedirs("model/saved_models", exist_ok=True)
results = []
roc_data = {}

# 3. Train, Save Model Files, and Evaluate All 6 Metrics
for model_name, clf in classifiers.items():
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    full_pipeline.fit(X_train, y_train)
    
    # Save trained model artifact
    file_slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = f"model/saved_models/{file_slug}.pkl"
    joblib.dump(full_pipeline, model_path)
    
    # Predictions on unseen test set
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": model_name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[model_name] = (fpr, tpr, auc)

results_df = pd.DataFrame(results)
results_df.to_csv("model/evaluation_results.csv", index=False)

print("\n=== MODEL EVALUATION COMPARISON TABLE ===")
print(results_df.to_string(index=False))

# 4. Visualization 1: Grouped Bar Chart Across All 6 Evaluation Metrics
melted_df = results_df.melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
ax = sns.barplot(data=melted_df, x="Metric", y="Score", hue="ML Model Name", palette="viridis")
plt.title("Performance Comparison of Classification Models Across 6 Metrics", fontsize=14, fontweight="bold")
plt.ylim(0, 1.12)
plt.ylabel("Metric Score", fontsize=11)
plt.xlabel("Evaluation Metric", fontsize=11)
plt.legend(title="Classification Model", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("model/metrics_comparison.png", dpi=300)
plt.close()

# 5. Visualization 2: ROC Curves Comparison
plt.figure(figsize=(8, 6))
for model_name, (fpr, tpr, auc_val) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.4f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Random Classifier (AUC = 0.5000)")
plt.title("ROC Curve Comparison Across All 5 Models", fontsize=13, fontweight="bold")
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model/roc_curves.png", dpi=300)
plt.close()

print("\nSaved trained models to model/saved_models/")
print("Saved visualizations: model/metrics_comparison.png and model/roc_curves.png")
