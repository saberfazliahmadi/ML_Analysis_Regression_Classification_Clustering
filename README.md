# ML Analysis: Regression, Classification, and Clustering

A comprehensive machine learning project that analyzes a tabular dataset using a full end-to-end workflow. The repository compares common regression and classification models, tunes hyperparameters, selects the best-performing model, and applies clustering techniques to uncover patterns in the data.

This project is designed as a practical and educational example of classical machine learning in Python, covering everything from data preprocessing to model evaluation and interpretation.

## Project Summary

This repository investigates a single dataset across multiple machine learning tasks:

- Regression modeling for continuous targets or performance comparison
- Classification modeling for predicting class labels
- Hyperparameter tuning with `GridSearchCV`
- Model selection based on validation performance
- Unsupervised clustering via K-Means++, Hierarchical Clustering, and DBSCAN

The goal is not only to train models, but also to compare them under the same conditions and understand which approach performs best for the dataset.

## What the Notebook Covers

The main notebook, `notebooks/a_comprehensive_ml_analysis.ipynb`, includes:

- dataset loading and inspection
- missing-value checks
- target variable identification
- train/test splitting
- feature scaling using `StandardScaler`
- regression model comparison
- classification model comparison
- cross-validation and hyperparameter tuning
- best-model selection
- clustering on selected features
- evaluation using relevant metrics

## Dataset

The project uses a preprocessed dataset located in:

- `data/preprocessed_dataset.csv`

The notebook is designed around a structured tabular dataset with a target column named `Class` and many encoded/engineered features. The data includes a mix of:

- numeric variables such as `PayloadMass`, `Flights`, `Block`, and related launch attributes
- binary or encoded categorical features
- target labels used for classification tasks

## Machine Learning Methods Included

### Regression Models

The workflow compares classical regression models such as:

- Linear Regression
- Decision Tree Regressor
- additional regression baselines used in the notebook

Typical regression metrics used include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Sum of Squared Errors (SSE)
- Sum of Absolute Errors (SAE)

### Classification Models

The project evaluates classical classification algorithms such as:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Typical classification metrics used include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve / AUC (when applicable)

### Clustering Models

The project also explores unsupervised clustering using:

- K-Means++
- Hierarchical Clustering
- DBSCAN

These methods are applied to selected feature pairs to identify natural groupings and patterns in the data.

## Workflow

The project follows a standard machine learning pipeline:

1. Load the dataset.
2. Inspect columns, row counts, and data types.
3. Check for missing values.
4. Split data into training and test sets.
5. Standardize features.
6. Train multiple models.
7. Compare model performance using metrics.
8. Use `GridSearchCV` for tuning.
9. Select the best model.
10. Apply clustering for exploratory data analysis.

## Repository Structure

```text
ML_Analysis_Regression_Classification_Clustering/
├── LICENSE
├── README.md
├── data/
│   └── preprocessed_dataset.csv
├── notebooks/
│   └── a_comprehensive_ml_analysis.ipynb
└── .gitignore (if present)
```

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/saberfazliahmadi/ML_Analysis_Regression_Classification_Clustering.git
cd ML_Analysis_Regression_Classification_Clustering
```

### 2) Open the notebook

The main analysis is in:

```text
notebooks/a_comprehensive_ml_analysis.ipynb
```

You can run it in:

- Jupyter Notebook
- JupyterLab
- Google Colab

### 3) Run the analysis

Follow the notebook prompts to provide:

- dataset path
- target variable name
- test size
- random state
- cross-validation value
- clustering feature names
- DBSCAN values like `min_samples` and `eps`

## Notes

- The notebook was originally designed with a Google Colab workflow in mind, including file access from a mounted Drive folder.
- It is easy to adapt for local execution by updating the dataset path and environment setup.
- This project is especially useful as a learning and demonstration tool for comparing classical ML models and understanding clustering behavior.

## Use Cases

This project is useful for:

- model comparison on the same dataset
- evaluating regression and classification pipelines
- learning hyperparameter tuning with cross-validation
- understanding clustering behavior in structured data
- demonstrating a practical end-to-end machine learning workflow

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Saber Fazliahmadi

GitHub: https://github.com/saberfazliahmadi

---

This repository is a practical machine learning analysis project focused on regression, classification, and clustering, with an emphasis on comparison, tuning, and interpreting model performance.
