import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Page Configuration

st.set_page_config(
    page_title="Machine Learning Explorer - Oswald",
    layout="wide"
)

st.title("Machine Learning Explorer")
st.write(
    """
    Explore supervised machine learning using a sample dataset or your own CSV file.
    Select a target, choose a model, adjust its settings, and evaluate its performance.
    """
)

# Load Dataset

@st.cache_data
def load_hazelnut_dataset():

    hazelnut = fetch_openml(
        data_id=46930,
        as_frame=True,
        target_column="Contaminated",
        parser="liac-arff"
    )

    df = hazelnut.data.copy()

    df["Contaminated"] = hazelnut.target

    return df

# Data source selection

st.header("1. Choose Your Dataset")

data_source = st.radio(
    "Select a data source:",
    [
        "Hazelnut Spread Contamination Detection (OpenML)",
        "Upload Your Own CSV"
    ],
    horizontal=True
)

df = None

if data_source == "Hazelnut Spread Contamination Detection (OpenML)":

    try:
        with st.spinner("Loading dataset from OpenML..."):
            df = load_hazelnut_dataset()

        st.success("Dataset loaded successfully!")

        st.caption(
            "Source: OpenML Dataset ID 46930"
        )

    except Exception as e:
        st.error(
            f"Unable to load the OpenML dataset. "
            f"Please check your internet connection or dataset ID. Error: {e}"
        )

else:

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV uploaded successfully!")

        except Exception as e:
            st.error(f"Unable to read the CSV file: {e}")


# Data Preview

if df is not None:

    st.header("2. Explore Your Data")

    st.write(
        f"Dataset contains **{df.shape[0]} rows** "
        f"and **{df.shape[1]} columns**."
    )

    with st.expander("Preview Dataset"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Dataset Information"):
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info_df, use_container_width=True)


    # Target and Feature Selection

    st.header("3. Select Your Target and Features")

    if data_source == "Hazelnut Spread Contamination Detection (OpenML)":

        if "Contaminated" not in df.columns:
            st.error(
                "The Hazelnut dataset loaded without its Contaminated target column. "
                "Please check the dataset-loading function."
            )
            st.stop()

        target_column = "Contaminated"

        st.info("The prediction target is automatically set to: Contaminated")

    else:

        target_column = st.selectbox(
            "Choose the column you want to predict:",
            options=df.columns.tolist()
        )

    # Remove rows where the selected target is missing.
    # This applies to both the Hazelnut dataset and uploaded CSV files.
    model_df = df.dropna(subset=[target_column]).copy()

    if model_df.empty:
        st.error("There are no rows with a valid target value.")
        st.stop()

    else:

        target_is_categorical = (
            model_df[target_column].dtype == "object"
            or isinstance(
                model_df[target_column].dtype,
                pd.CategoricalDtype
            )
            or model_df[target_column].dtype == "bool"
        )

        if not target_is_categorical:
            target_is_categorical = (
                model_df[target_column].nunique() <= 20
            )

        task_type = (
            "Classification"
            if target_is_categorical
            else "Regression"
        )

        st.info(f"Detected task type: **{task_type}**")

        X = model_df.drop(columns=[target_column]).copy()
        y = model_df[target_column].copy()

        for column in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[column]):
                X[column] = X[column].astype("int64")

        usable_columns = [
            column for column in X.columns
            if X[column].nunique(dropna=True) > 1
        ]

        X = X[usable_columns]

        if X.empty:
            st.error("No usable feature columns were found.")

        else:

            selected_features = st.multiselect(
                "Choose the input features for your model:",
                options=list(X.columns),
                default=list(X.columns)
            )

            if not selected_features:
                st.warning("Please select at least one input feature.")

            else:

                X = X[selected_features].copy()

                if task_type == "Classification":
                    y = y.astype(str)

                else:
                    y = pd.to_numeric(y, errors="coerce")

                    valid_rows = y.notna()
                    X = X.loc[valid_rows]
                    y = y.loc[valid_rows]

                if len(X) < 5:
                    st.error(
                        "The dataset needs at least five valid rows "
                        "to train a model."
                    )

                else:

                    # Train/Test Split Configuration

                    st.header("4. Configure Training")

                    col1, col2 = st.columns(2)

                    with col1:
                        test_size = st.slider(
                            "Test set size (%)",
                            min_value=10,
                            max_value=40,
                            value=20,
                            step=5
                        )

                    with col2:
                        random_state = st.number_input(
                            "Random state",
                            min_value=0,
                            max_value=10000,
                            value=42
                        )

                    # Model Selection

                    st.header("5. Choose a Machine Learning Model")

                    if task_type == "Classification":

                        model_name = st.selectbox(
                            "Select a classification model:",
                            [
                                "Logistic Regression",
                                "K-Nearest Neighbors",
                                "Decision Tree",
                                "Random Forest"
                            ]
                        )

                    else:

                        model_name = st.selectbox(
                            "Select a regression model:",
                            [
                                "Linear Regression",
                                "Decision Tree",
                                "Random Forest"
                            ]
                        )


                    # Hyperparameter Settings

                    st.subheader("Adjust Model Hyperparameters")

                    if model_name == "Linear Regression":

                        fit_intercept = st.checkbox(
                            "Fit intercept",
                            value=True
                        )

                        model = LinearRegression(
                            fit_intercept=fit_intercept
                        )

                    elif model_name == "Logistic Regression":

                        col1, col2 = st.columns(2)

                        with col1:
                            C = st.slider(
                                "Regularization strength (C)",
                                min_value=0.01,
                                max_value=10.0,
                                value=1.0,
                                step=0.1
                            )

                        with col2:
                            max_iter = st.slider(
                                "Maximum iterations",
                                min_value=100,
                                max_value=2000,
                                value=1000,
                                step=100
                            )

                        model = LogisticRegression(
                            C=C,
                            max_iter=max_iter
                        )

                    elif model_name == "K-Nearest Neighbors":

                        col1, col2 = st.columns(2)

                        with col1:
                            n_neighbors = st.slider(
                                "Number of neighbors",
                                min_value=1,
                                max_value=20,
                                value=5
                            )

                        with col2:
                            weights = st.selectbox(
                                "Neighbor weights",
                                ["uniform", "distance"]
                            )

                        model = KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights=weights
                        )

                    elif model_name == "Decision Tree":

                        col1, col2 = st.columns(2)

                        with col1:
                            max_depth = st.slider(
                                "Maximum tree depth",
                                min_value=1,
                                max_value=20,
                                value=5
                            )

                        with col2:
                            min_samples_split = st.slider(
                                "Minimum samples to split",
                                min_value=2,
                                max_value=20,
                                value=2
                            )

                        if task_type == "Classification":

                            criterion = st.selectbox(
                                "Splitting criterion",
                                ["gini", "entropy", "log_loss"]
                            )

                            model = DecisionTreeClassifier(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                criterion=criterion,
                                random_state=random_state
                            )

                        else:

                            criterion = st.selectbox(
                                "Splitting criterion",
                                [
                                    "squared_error",
                                    "absolute_error",
                                    "friedman_mse"
                                ]
                            )

                            model = DecisionTreeRegressor(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                criterion=criterion,
                                random_state=random_state
                            )

                    elif model_name == "Random Forest":

                        col1, col2 = st.columns(2)

                        with col1:
                            n_estimators = st.slider(
                                "Number of trees",
                                min_value=10,
                                max_value=300,
                                value=100,
                                step=10
                            )

                        with col2:
                            max_depth = st.slider(
                                "Maximum tree depth",
                                min_value=1,
                                max_value=20,
                                value=10
                            )

                        min_samples_split = st.slider(
                            "Minimum samples to split",
                            min_value=2,
                            max_value=20,
                            value=2
                        )

                        if task_type == "Classification":

                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=random_state
                            )

                        else:

                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=random_state
                            )


                    # Data Preprocessing Pipelines

                    numeric_features = X.select_dtypes(
                        include=np.number
                    ).columns.tolist()

                    categorical_features = X.select_dtypes(
                        exclude=np.number
                    ).columns.tolist()

                    numeric_transformer = Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="median")
                            ),
                            (
                                "scaler",
                                StandardScaler()
                            )
                        ]
                    )

                    categorical_transformer = Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="most_frequent")
                            ),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore")
                            )
                        ]
                    )

                    transformers = []

                    if numeric_features:
                        transformers.append(
                            (
                                "numeric",
                                numeric_transformer,
                                numeric_features
                            )
                        )

                    if categorical_features:
                        transformers.append(
                            (
                                "categorical",
                                categorical_transformer,
                                categorical_features
                            )
                        )

                    preprocessor = ColumnTransformer(
                        transformers=transformers
                    )

                    # Training and Evaluation

                    st.header("6. Model Results")

                    try:

                        stratify_values = None

                        if task_type == "Classification":
                            class_counts = y.value_counts()

                            if (
                                len(class_counts) > 1
                                and class_counts.min() >= 2
                            ):
                                stratify_values = y

                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=test_size / 100,
                            random_state=int(random_state),
                            stratify=stratify_values
                        )

                        # Build a complete pipeline
                        pipeline = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("model", model)
                            ]
                        )

                        # Train automatically whenever settings change
                        with st.spinner("Training your model..."):

                            pipeline.fit(X_train, y_train)

                            predictions = pipeline.predict(X_test)

                        st.success("Model trained successfully!")

                        st.write(
                            f"Training observations: **{len(X_train)}**"
                        )

                        st.write(
                            f"Testing observations: **{len(X_test)}**"
                        )

                        # Classification Metrics

                        if task_type == "Classification":

                            accuracy = accuracy_score(
                                y_test,
                                predictions
                            )

                            precision = precision_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0
                            )

                            recall = recall_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0
                            )

                            f1 = f1_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0
                            )

                            metric1, metric2, metric3, metric4 = st.columns(4)

                            metric1.metric(
                                "Accuracy",
                                f"{accuracy:.3f}"
                            )

                            metric2.metric(
                                "Precision",
                                f"{precision:.3f}"
                            )

                            metric3.metric(
                                "Recall",
                                f"{recall:.3f}"
                            )

                            metric4.metric(
                                "F1 Score",
                                f"{f1:.3f}"
                            )

                            st.subheader("Understanding the Metrics")

                            st.write(
                                """
                                - **Accuracy:** The percentage of predictions
                                  that were correct.
                                - **Precision:** Of the observations predicted
                                  as a particular class, how many were correct.
                                - **Recall:** Of the observations that actually
                                  belonged to a class, how many the model found.
                                - **F1 Score:** A balance between precision
                                  and recall. Higher values generally indicate
                                  better performance.
                                """
                            )

                            st.subheader("Confusion Matrix")

                            fig, ax = plt.subplots()

                            ConfusionMatrixDisplay.from_predictions(
                                y_test,
                                predictions,
                                ax=ax,
                                xticks_rotation="vertical"
                            )

                            ax.set_title("Confusion Matrix")

                            st.pyplot(fig)
                            plt.close(fig)


                        # Regression Metrics

                        else:

                            r2 = r2_score(
                                y_test,
                                predictions
                            )

                            mae = mean_absolute_error(
                                y_test,
                                predictions
                            )

                            rmse = np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    predictions
                                )
                            )

                            metric1, metric2, metric3 = st.columns(3)

                            metric1.metric(
                                "R² Score",
                                f"{r2:.3f}"
                            )

                            metric2.metric(
                                "Mean Absolute Error",
                                f"{mae:.3f}"
                            )

                            metric3.metric(
                                "Root Mean Squared Error",
                                f"{rmse:.3f}"
                            )

                            st.subheader("Understanding the Metrics")

                            st.write(
                                """
                                - **R² Score:** Measures how much of the
                                  variation in the target is explained by
                                  the model. A score closer to 1 is generally
                                  better.
                                - **Mean Absolute Error (MAE):** The average
                                  absolute difference between actual and
                                  predicted values. Lower is better.
                                - **Root Mean Squared Error (RMSE):** Measures
                                  prediction error while penalizing larger
                                  errors more heavily. Lower is better.
                                """
                            )

                            # Actual vs. predicted chart
                            st.subheader("Actual vs. Predicted Values")

                            fig, ax = plt.subplots()

                            ax.scatter(
                                y_test,
                                predictions,
                                alpha=0.7
                            )

                            min_value = min(
                                y_test.min(),
                                predictions.min()
                            )

                            max_value = max(
                                y_test.max(),
                                predictions.max()
                            )

                            ax.plot(
                                [min_value, max_value],
                                [min_value, max_value],
                                linestyle="--"
                            )

                            ax.set_xlabel("Actual Values")
                            ax.set_ylabel("Predicted Values")
                            ax.set_title("Actual vs. Predicted")

                            st.pyplot(fig)
                            plt.close(fig)

                            # Residual plot.
                            st.subheader("Residual Plot")

                            residuals = y_test - predictions

                            fig, ax = plt.subplots()

                            ax.scatter(
                                predictions,
                                residuals,
                                alpha=0.7
                            )

                            ax.axhline(
                                y=0,
                                linestyle="--"
                            )

                            ax.set_xlabel("Predicted Values")
                            ax.set_ylabel("Residuals")
                            ax.set_title("Prediction Errors")

                            st.pyplot(fig)
                            plt.close(fig)


                        # Prediction Preview

                        st.subheader("Prediction Preview")

                        results = pd.DataFrame({
                            "Actual Value": y_test.values,
                            "Predicted Value": predictions
                        })

                        st.dataframe(
                            results.head(20),
                            use_container_width=True
                        )

                    except Exception as e:

                        st.error(
                            f"An error occurred while training the model: {e}"
                        )