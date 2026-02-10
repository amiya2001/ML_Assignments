import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier

st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

st.title("🚨 Network Intrusion Detection System")
st.write("Machine Learning based Intrusion Detection using UNSW-NB15 dataset")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV file (UNSW-NB15 format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------
    # Preprocessing
    # ----------------------------
    target_column = "label"
    drop_columns = ["id", "attack_cat"]

    X = df.drop(columns=drop_columns + [target_column], errors="ignore")
    y = df[target_column]

    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore",sparse_output=False), cat_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ----------------------------
    # Model Selection
    # ----------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        [
            "Logistic Regression",
            "KNN",
            "Naive Bayes",
            "Random Forest","XGBoost",
            "SVM"
            
        ],
    )

    if model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_name == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        classifier = GaussianNB()
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "SVM":
        classifier = SVC(kernel="linear", probability=True, random_state=42)
    else:
        classifier =XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    if st.button("Train & Evaluate"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1-score:** {f1:.4f}")

        cm = confusion_matrix(y_test, y_pred)

        st.subheader("Confusion Matrix")
        st.write(cm)