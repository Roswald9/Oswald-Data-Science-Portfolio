
# Machine Learning Explorer: Hazelnut Contamination Detection

## Project Overview

This project is an interactive machine learning application developed using Python, Streamlit, and Scikit-learn. Users can explore supervised machine learning by training models on a sample dataset or uploading their own CSV files.

The sample dataset is the Hazelnut Contamination Detection Dataset from OpenML. The objective is to predict whether a hazelnut spread sample is contaminated based on its microwave measurement characteristics.

Users can select prediction targets, choose machine learning algorithms, adjust hyperparameters, and evaluate model performance through interactive metrics and visualizations.

## Dataset

**Source:** [OpenML - Hazelnut Contamination Detection](https://www.openml.org/d/46930)

- Observations: 2,400
- Input measurements: 30
- Target variable: `Contaminated`
- Problem type: Binary Classification

The application also allows users to upload their own CSV datasets for additional machine learning experiments.

## Application Features

- **Dataset Selection:** Load the OpenML hazelnut dataset or upload a custom CSV file.
- **Target Selection:** Choose the variable the model will predict.
- **Feature Selection:** Select which input variables will be used for training.
- **Model Selection:** Choose from several supervised machine learning algorithms.
- **Hyperparameter Adjustment:** Modify model settings and automatically retrain models.
- **Performance Evaluation:** Compare model performance using multiple evaluation metrics.
- **Data Visualization:** View confusion matrices and prediction results.

## Machine Learning Models

The application supports the following algorithms:

| Model | Adjustable Hyperparameters |
|---|---|
| Logistic Regression | Regularization strength (C), maximum iterations |
| K-Nearest Neighbors | Number of neighbors, neighbor weights |
| Decision Tree | Maximum depth, minimum samples to split, splitting criterion |
| Random Forest | Number of trees, maximum depth, minimum samples to split |
| Linear Regression | Fit intercept (regression tasks) |

Regression datasets also support Decision Tree Regression and Random Forest Regression.

## Model Evaluation

For classification tasks, the application calculates:

- **Accuracy:** Percentage of predictions that are correct.
- **Precision:** Percentage of positive predictions that are correct.
- **Recall:** Percentage of actual positive cases correctly identified.
- **F1 Score:** Balance between precision and recall.

For regression tasks, the application calculates:

- **R² Score:** Measures how much variation in the target is explained by the model.
- **Mean Absolute Error (MAE):** Average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors.

## Technologies and Libraries

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Running the Application Locally

1. Clone or download the GitHub repository.
2. Navigate to the `MLStreamlitApp` directory.
3. Install the required dependencies:
4. Launch the application:
5. Open the local URL displayed in the terminal.

## References

1. [OpenML - Hazelnut Contamination Detection Dataset](https://www.openml.org/d/46930)
2. [Streamlit Documentation](https://docs.streamlit.io/)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [Pandas Documentation](https://pandas.pydata.org/docs/)

---

Developed as part of a Data Science portfolio project to demonstrate supervised machine learning, model evaluation, Python programming, and interactive application development.
