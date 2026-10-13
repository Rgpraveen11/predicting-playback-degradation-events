from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "data" /
    "media_service_assurance_live_qoe_50k.csv"
)

TARGET = "Playback_Degradation_Event"
ID_COL = "Session_ID"

st.set_page_config(
    page_title="Playback Degradation Predictor",
    page_icon="📺",
    layout="wide",
)

st.title("📺 Predicting Playback Degradation Events")
st.caption(
    "Machine learning dashboard for media service assurance, "
    "playback risk prediction, and model explainability."
)

# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)

    if TARGET not in data.columns:
        raise ValueError(
            f"Required target column '{TARGET}' is missing."
        )

    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = pd.to_numeric(
        data[TARGET], errors="coerce"
    )
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = data[TARGET].astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Playback_Degradation_Event must contain 0 and 1."
        )

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "The dataset must contain both target classes."
        )

    return data


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Place the CSV file inside the project's data folder."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# Remove identifier and target from predictive inputs.
X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
y = df[TARGET]

if X.shape[1] == 0:
    st.error("No input features are available.")
    st.stop()

# ==================================================
# 3. PREPROCESSING AND MODEL TRAINING
# ==================================================

@st.cache_resource
def train_and_compare_models(data):
    X_all = data.drop(
        columns=[TARGET, ID_COL], errors="ignore"
    )
    y_all = data[TARGET].astype(int)

    numeric_cols = X_all.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X_all.select_dtypes(
        include=["object", "category", "bool", "str"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all,
    )

    algorithms = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    trained_models = {}
    rows = []

    for name, classifier in algorithms.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        # Save the trained Random Forest pipeline
        if name == "Random Forest":
            import joblib
            from pathlib import Path

            model_dir = Path(__file__).resolve().parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(
                pipeline,
                model_dir / "trained_model.joblib"
            )

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        scores = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }

        rows.append(scores)

        trained_models[name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    comparison = pd.DataFrame(rows).sort_values(
        "F1 Score", ascending=False
    ).reset_index(drop=True)

    # Choose the highest F1 score as the default model-selection
    # rule. Other criteria may be more suitable in production.
    best_name = comparison.iloc[0]["Model"]
    best = trained_models[best_name]

    return {
        "models": trained_models,
        "comparison": comparison,
        "best_name": best_name,
        "best_pipeline": best["pipeline"],
        "best_predictions": best["predictions"],
        "best_probabilities": best["probabilities"],
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": X_all.columns.tolist(),
    }


with st.spinner(
    "Preprocessing data and comparing three ML models..."
):
    try:
        results = train_and_compare_models(df)
    except Exception as error:
        st.error(f"Model training failed: {error}")
        st.stop()

# ==================================================
# 4. DASHBOARD NAVIGATION
# ==================================================

overview_tab, evaluation_tab, prediction_tab, dataset_tab, importance_tab, shap_tab = (
    st.tabs([
        "Overview",
        "Model evaluation",
        "Predict a session",
        "Dataset",
        "Feature importance",
        "SHAP explainability",
    ])
)

# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with overview_tab:
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total sessions", f"{len(df):,}")
    c2.metric("Input features", len(results["feature_names"]))
    c3.metric(
        "Degradation events",
        f"{int((df[TARGET] == 1).sum()):,}",
    )
    c4.metric(
        "Event rate",
        f"{df[TARGET].mean() * 100:.2f}%",
    )

    st.success(
        f"Selected model: {results['best_name']} "
        "based on test-set F1 score."
    )

    left, right = st.columns(2)

    with left:
        st.write("**Playback event distribution**")

        counts = df[TARGET].value_counts().reindex(
            [0, 1], fill_value=0
        )

        fig, ax = plt.subplots()
        ax.bar(
            ["Smooth (0)", "Degradation (1)"],
            counts.values,
        )
        ax.set_ylabel("Number of sessions")
        ax.set_title("Target distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.write("**Numeric feature correlations**")

        numeric_df = df.select_dtypes(
            include="number"
        ).drop(columns=[ID_COL], errors="ignore")

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(
            corr, aspect="auto", vmin=-1, vmax=1,
            cmap="coolwarm"
        )

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(
            corr.columns, rotation=75,
            ha="right", fontsize=7
        )
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=7)
        ax.set_title("Correlation heatmap")

        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==================================================
# TAB 2: MODEL EVALUATION
# ==================================================

with evaluation_tab:
    st.subheader("Compare machine learning algorithms")

    st.write(
        "All models use the same stratified train-test split. "
        "The best model is selected by F1 score by default."
    )

    comparison = results["comparison"]

    st.dataframe(
        comparison.round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.bar_chart(
        comparison.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        ]
    )

    st.success(
        f"Current selected model: {results['best_name']}"
    )

    st.divider()
    st.subheader("Selected model test results")

    metric_cols = st.columns(5)

    selected_row = comparison[
        comparison["Model"] == results["best_name"]
    ].iloc[0]

    metric_keys = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1 score", "F1 Score"),
        ("ROC-AUC", "ROC-AUC"),
    ]

    for col, (label, key) in zip(metric_cols, metric_keys):
        col.metric(label, f"{selected_row[key]:.3f}")

    cm = confusion_matrix(
        results["y_test"],
        results["best_predictions"],
        labels=[0, 1],
    )

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted smooth", "Predicted event"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual smooth", "Actual event"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center"
            )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        results["y_test"],
        results["best_predictions"],
        labels=[0, 1],
        target_names=["Smooth (0)", "Degradation (1)"],
        output_dict=True,
        zero_division=0,
    )

    st.write("**Classification report**")
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True,
    )

    st.download_button(
        "Download model comparison",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# ==================================================
# TAB 3: PREDICT A SESSION
# ==================================================

with prediction_tab:
    st.subheader("Predict playback degradation risk")

    st.write(
        "Enter the measurements for a hypothetical streaming session."
    )

    expected = [
        "Mobile_Carrier_Network",
        "Signal_Strength_dBm",
        "Cell_Tower_Congestion_Ratio",
        "Concurrent_Regional_Viewers_k",
        "CDN_Edge_Latency_ms",
        "Requested_Bitrate_Mbps",
        "Available_Bandwidth_Mbps",
        "Device_CPU_Load_Pct",
    ]

    missing = [
        col for col in expected
        if col not in results["feature_names"]
    ]

    if missing:
        st.error(
            "Prediction form columns do not match the dataset: "
            + ", ".join(missing)
        )
    else:
        with st.form("prediction_form"):
            left, right = st.columns(2)

            carriers = sorted(
                df["Mobile_Carrier_Network"]
                .dropna().astype(str).unique().tolist()
            )

            with left:
                carrier = st.selectbox(
                    "Mobile carrier network", carriers
                )
                signal = st.slider(
                    "Signal strength (dBm)",
                    -120, -50, -85
                )
                congestion = st.slider(
                    "Cell tower congestion ratio",
                    0.0, 1.5, 0.70, 0.05
                )
                viewers = st.number_input(
                    "Concurrent regional viewers (thousands)",
                    min_value=0.0, max_value=2000.0,
                    value=300.0, step=10.0
                )

            with right:
                latency = st.number_input(
                    "CDN edge latency (ms)",
                    min_value=0, max_value=1000,
                    value=80, step=5
                )
                bitrate = st.number_input(
                    "Requested bitrate (Mbps)",
                    min_value=0.1, max_value=100.0,
                    value=8.0, step=0.5
                )
                bandwidth = st.number_input(
                    "Available bandwidth (Mbps)",
                    min_value=0.1, max_value=200.0,
                    value=12.0, step=0.5
                )
                cpu = st.slider(
                    "Device CPU load (%)",
                    0.0, 100.0, 55.0, 1.0
                )

            submitted = st.form_submit_button(
                "Predict degradation risk",
                type="primary",
            )

        if submitted:
            sample = pd.DataFrame([{
                "Mobile_Carrier_Network": carrier,
                "Signal_Strength_dBm": signal,
                "Cell_Tower_Congestion_Ratio": congestion,
                "Concurrent_Regional_Viewers_k": viewers,
                "CDN_Edge_Latency_ms": latency,
                "Requested_Bitrate_Mbps": bitrate,
                "Available_Bandwidth_Mbps": bandwidth,
                "Device_CPU_Load_Pct": cpu,
            }])

            sample = sample[results["feature_names"]]

            best_model = results["best_pipeline"]
            probability = float(
                best_model.predict_proba(sample)[0, 1]
            )

            prediction = int(probability >= 0.5)

            st.metric(
                "Estimated degradation probability",
                f"{probability * 100:.1f}%",
            )

            if prediction == 1:
                st.warning(
                    "Higher risk: the model predicts a "
                    "playback degradation event."
                )
            else:
                st.success(
                    "Lower risk: the model predicts "
                    "smooth playback."
                )

            st.caption(
                "The 50% threshold is a default, not a guarantee. "
                "Tune it using validation data for the intended use."
            )

# ==================================================
# TAB 4: DATASET EXPLORER
# ==================================================

with dataset_tab:
    st.subheader("Dataset explorer")

    st.write("**First 100 records**")
    st.dataframe(df.head(100), use_container_width=True)

    st.write("**Column data types**")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.write("**Missing values per column**")
    missing_df = df.isna().sum().rename(
        "Missing values"
    ).to_frame()
    st.dataframe(missing_df, use_container_width=True)

    st.download_button(
        "Download dataset preview",
        data=df.head(100).to_csv(index=False).encode("utf-8"),
        file_name="playback_degradation_preview.csv",
        mime="text/csv",
    )

# ==================================================
# TAB 5: FEATURE IMPORTANCE
# ==================================================

with importance_tab:
    st.subheader("Random Forest feature importance")

    # Explain the Random Forest model specifically.
    rf_model = results["models"]["Random Forest"]["pipeline"]
    rf_classifier = rf_model.named_steps["classifier"]
    rf_preprocessor = rf_model.named_steps["preprocess"]

    transformed_names = rf_preprocessor.get_feature_names_out()
    raw_importance = rf_classifier.feature_importances_

    aggregated = {
        name: 0.0 for name in results["feature_names"]
    }

    for transformed_name, score in zip(
        transformed_names, raw_importance
    ):
        if transformed_name.startswith("numeric__"):
            original = transformed_name[len("numeric__"):]

        elif transformed_name.startswith("categorical__"):
            encoded = transformed_name[len("categorical__"):]
            matches = [
                col for col in results["categorical_cols"]
                if encoded == col or encoded.startswith(col + "_")
            ]
            original = max(matches, key=len) if matches else encoded

        else:
            original = transformed_name

        if original in aggregated:
            aggregated[original] += float(score)

    importance_df = pd.DataFrame({
        "Feature": list(aggregated.keys()),
        "Importance": list(aggregated.values()),
    }).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)

    st.info(
        "These values indicate model reliance, not proof of causation."
    )

    top_n = st.slider(
        "Features to display",
        min_value=1,
        max_value=len(importance_df),
        value=min(8, len(importance_df)),
        key="importance_top_n",
    )

    plot_df = importance_df.head(top_n).sort_values(
        "Importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["Feature"], plot_df["Importance"])
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Feature importance ranking")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        importance_df.round(5),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download feature importance CSV",
        data=importance_df.to_csv(index=False).encode("utf-8"),
        file_name="feature_importance.csv",
        mime="text/csv",
    )

# ==================================================
# TAB 6: SHAP EXPLAINABILITY
# ==================================================

with shap_tab:
    st.subheader("SHAP model explainability")

    st.write(
        "SHAP estimates how features contribute to predictions. "
        "This section explains the Random Forest model."
    )

    try:
        import shap

        rf_model = results["models"]["Random Forest"]["pipeline"]
        rf_classifier = rf_model.named_steps["classifier"]
        rf_preprocessor = rf_model.named_steps["preprocess"]

        # Limit sample size to keep the explanation responsive.
        X_explain = results["X_test"].head(100)
        X_transformed = rf_preprocessor.transform(X_explain)

        explainer = shap.TreeExplainer(rf_classifier)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            event_values = shap_values[1]
        elif len(np.asarray(shap_values).shape) == 3:
            event_values = np.asarray(shap_values)[:, :, 1]
        else:
            event_values = np.asarray(shap_values)

        transformed_names = rf_preprocessor.get_feature_names_out()

        shap_importance = pd.DataFrame({
            "Feature": transformed_names,
            "Mean absolute SHAP value": np.abs(
                event_values
            ).mean(axis=0),
        }).sort_values(
            "Mean absolute SHAP value",
            ascending=False,
        )

        st.caption(
            "Mean absolute SHAP values summarize feature influence "
            "across the selected test sessions."
        )

        st.bar_chart(
            shap_importance.head(15).set_index("Feature")
        )

        st.dataframe(
            shap_importance.head(15).round(5),
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download SHAP importance CSV",
            data=shap_importance.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="shap_importance.csv",
            mime="text/csv",
        )

    except ImportError:
        st.warning(
            "SHAP is not installed. Install it using: "
            "python -m pip install shap"
        )
    except Exception as error:
        st.error(f"Could not calculate SHAP values: {error}")

# ==================================================
# PROJECT NOTES
# ==================================================

with st.expander("Project methodology and limitations"):
    st.markdown("""
    **Methodology**
    1. Load and validate the playback session dataset.
    2. Impute missing values and encode categorical features.
    3. Train Logistic Regression, Random Forest and Gradient Boosting.
    4. Compare models on the same held-out test set.
    5. Select the model with the highest F1 score as a baseline.
    6. Visualize model performance and feature importance.
    7. Use SHAP to analyze Random Forest predictions.

    **Limitations**
    - A high score on one split does not guarantee real-world performance.
    - Class imbalance makes precision, recall and the confusion matrix important.
    - Feature importance does not establish causality.
    - The default 50% threshold should be evaluated for the intended use.
    - This application currently runs as a Streamlit app. FastAPI,
      automated tests, cloud deployment and a saved model pipeline
      require separate implementation.
    """)