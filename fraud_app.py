import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

st.set_page_config(page_title="Secure Bank Fraud Detection", layout="wide")

# =====================================================
# 1. CONSTANTS AND HELPERS
# =====================================================

FEATURE_COLS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

LABEL_COL = "fraud"


@st.cache_data
def load_default_kaggle_data():
    """
    Load the default Kaggle dataset from card_transdata.csv
    The file should be in the same folder as this app.
    """
    df = pd.read_csv("card_transdata.csv")
    return df


def train_fraud_model(df: pd.DataFrame):
    """
    Train a Random Forest on the given dataset.
    Returns model and train test splits for evaluation.
    """
    # Basic checks
    if LABEL_COL not in df.columns:
        st.error(f"Label column '{LABEL_COL}' not found in data.")
        st.stop()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    # If only one class present, model cannot be trained for classification
    if len(np.unique(y)) < 2:
        st.error("Label column has only one class. Cannot train a fraud model.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# =====================================================
# 2. SIDEBAR CONFIGURATION
# =====================================================

st.sidebar.header("Configuration")

data_mode = st.sidebar.radio(
    "Choose data source",
    ["Use default Kaggle sample", "Upload my own CSV"],
)

uploaded_file = None
user_df = None

if data_mode == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV with transaction data",
        type="csv",
        help=(
            "Expected columns: distance_from_home, distance_from_last_transaction, "
            "ratio_to_median_purchase_price, repeat_retailer, used_chip, "
            "used_pin_number, online_order, and fraud."
        ),
    )
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

risk_threshold = st.sidebar.slider(
    "Fraud alert threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="If predicted fraud probability is above this value, the transaction is flagged as fraud.",
)

# =====================================================
# 3. LOAD DATA AND TRAIN MODEL
# =====================================================

if user_df is not None:
    data = user_df.copy()
    st.success(f"Using uploaded dataset with {len(data)} rows.")
else:
    try:
        data = load_default_kaggle_data()
        st.info(f"Using default Kaggle dataset with {len(data)} rows.")
    except FileNotFoundError:
        st.error(
            "Default Kaggle dataset 'card_transdata.csv' not found. "
            "Place it in the same folder as this app or upload a CSV instead."
        )
        st.stop()

model, X_train, X_test, y_train, y_test = train_fraud_model(data)

# Compute probabilities and predictions on test set
y_proba_test = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_proba_test >= risk_threshold).astype(int)

# Metrics
cm = confusion_matrix(y_test, y_pred_default)
report_dict = classification_report(y_test, y_pred_default, output_dict=True)
fraud_precision = report_dict.get("1", {}).get("precision", 0.0)
fraud_recall = report_dict.get("1", {}).get("recall", 0.0)
overall_accuracy = report_dict.get("accuracy", 0.0)

fpr, tpr, _ = roc_curve(y_test, y_proba_test)
roc_auc = auc(fpr, tpr)

# =====================================================
# 4. PAGE HEADER
# =====================================================

st.title("Secure Bank: Card Transaction Fraud Detection")

st.markdown(
    """
This application demonstrates an AI system that detects **credit card transaction fraud**  
using a real Kaggle dataset.  

You can:
- Use the default Kaggle sample
- Upload your own file in the same format
- Adjust the fraud threshold to see how it affects detection
"""
)

# =====================================================
# 5. METRICS AND VISUALS (MODEL LEVEL)
# =====================================================

st.subheader("Model performance on test data")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric("Accuracy", f"{overall_accuracy:.3f}")

with col_m2:
    st.metric("Fraud recall (class 1)", f"{fraud_recall:.3f}")

with col_m3:
    st.metric("Fraud precision (class 1)", f"{fraud_precision:.3f}")

with col_m4:
    st.metric("ROC AUC", f"{roc_auc:.3f}")

col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("**Confusion matrix (test set)**")
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax_cm.set_yticklabels(["Legit (0)", "Fraud (1)"])
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)

with col_v2:
    st.markdown("**ROC curve (test set)**")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

st.subheader("Class distribution and feature importance")

col_d1, col_d2 = st.columns(2)

with col_d1:
    class_counts = data[LABEL_COL].value_counts().sort_index()
    fig_dist, ax_dist = plt.subplots()
    ax_dist.bar(["Legit (0)", "Fraud (1)"], class_counts.values)
    ax_dist.set_ylabel("Number of transactions")
    ax_dist.set_title("Class distribution in dataset")
    st.pyplot(fig_dist)

with col_d2:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [FEATURE_COLS[i] for i in indices]

    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(range(len(importances)), importances[indices])
    ax_imp.set_xticks(range(len(importances)))
    ax_imp.set_xticklabels(feature_names, rotation=45, ha="right")
    ax_imp.set_ylabel("Importance")
    ax_imp.set_title("Feature importance (Random Forest)")
    fig_imp.tight_layout()
    st.pyplot(fig_imp)

# =====================================================
# 6. SINGLE TRANSACTION PREDICTION
# =====================================================

st.subheader("Test a single transaction")

c1, c2, c3, c4 = st.columns(4)

with c1:
    distance_from_home = st.number_input(
        "Distance from home",
        min_value=0.0,
        value=10.0,
    )
    repeat_retailer = st.selectbox(
        "Repeat retailer",
        ["No", "Yes"],
    )

with c2:
    distance_from_last = st.number_input(
        "Distance from last transaction",
        min_value=0.0,
        value=1.0,
    )
    used_chip = st.selectbox(
        "Used chip",
        ["No", "Yes"],
    )

with c3:
    ratio_to_median = st.number_input(
        "Ratio to median purchase price",
        min_value=0.0,
        value=1.0,
    )
    used_pin_number = st.selectbox(
        "Used PIN",
        ["No", "Yes"],
    )

with c4:
    online_order = st.selectbox(
        "Online order",
        ["No", "Yes"],
    )

if st.button("Predict fraud risk for this transaction"):
    input_row = pd.DataFrame(
        [
            {
                "distance_from_home": distance_from_home,
                "distance_from_last_transaction": distance_from_last,
                "ratio_to_median_purchase_price": ratio_to_median,
                "repeat_retailer": 1 if repeat_retailer == "Yes" else 0,
                "used_chip": 1 if used_chip == "Yes" else 0,
                "used_pin_number": 1 if used_pin_number == "Yes" else 0,
                "online_order": 1 if online_order == "Yes" else 0,
            }
        ]
    )

    prob = model.predict_proba(input_row)[:, 1][0]
    label = "Fraud" if prob >= risk_threshold else "Legit"

    st.markdown("### Prediction result")
    st.write(f"Estimated fraud probability: **{prob:.3f}**")
    st.write(f"Decision at threshold {risk_threshold:.2f}: **{label}**")

# =====================================================
# 7. SCORE FULL DATASET AND DOWNLOAD
# =====================================================

st.subheader("Score the entire dataset")

st.markdown(
    """
Use this section to score every row in the current dataset  
(default Kaggle sample or your uploaded file) and download the results.
"""
)

if st.button("Score all transactions"):
    X_all = data[FEATURE_COLS]
    probs_all = model.predict_proba(X_all)[:, 1]
    preds_all = (probs_all >= risk_threshold).astype(int)

    result_df = data.copy()
    result_df["fraud_probability"] = probs_all
    result_df["fraud_prediction"] = preds_all

    st.write("Preview of scored data (first 50 rows):")
    st.dataframe(result_df.head(50))

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored dataset as CSV",
        data=csv_bytes,
        file_name="scored_card_transactions.csv",
        mime="text/csv",
    )
