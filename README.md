# ML Analysis: Regression, Classification, and Clustering

A complete machine learning project that evaluates a dataset using a combination of supervised and unsupervised learning techniques. The workflow compares common regression models, evaluates classification algorithms, tunes hyperparameters with GridSearchCV, and applies clustering methods to uncover patterns in the data.

This repository demonstrates a practical end-to-end machine learning pipeline in Python and is designed to be easy to understand, reuse, and extend.

## Overview

This project covers the full analysis lifecycle:

- loading and inspecting a tabular dataset
- identifying the target variable and feature columns
- splitting data into training and test sets
- scaling feature values for model training
- comparing regression models
- comparing classification models
- selecting the best model using cross-validation and hyperparameter tuning
- applying clustering methods to analyze structure in the data
- interpreting model performance using standard evaluation metrics

## Project Goals

The main goals of this project are to:

1. Evaluate multiple machine learning models on the same dataset.
2. Compare regression and classification performance.
3. Identify the strongest model using systematic tuning and validation.
4. Explore clustering patterns in feature space.
5. Present a clear and reusable workflow for educational and experimental ML analysis.

## Methods Included

### Regression analysis

The notebook compares common regression approaches such as:

- Linear Regression
- Decision Tree Regressor
- other classical regression baselines used in the workflow

Evaluation metrics include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Sum of Squared Errors (SSE)
- Sum of Absolute Errors (SAE)

### Classification analysis

The workflow compares a range of classic classification algorithms, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Common evaluation metrics used in the project:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and AUC (when applicable)

### Clustering analysis

The project also includes unsupervised clustering techniques:

- K-Means++
- Hierarchical Clustering
- DBSCAN

These methods are used to investigate naturally occurring groupings in the dataset based on selected feature pairs.

## Dataset

The project uses the preprocessed dataset stored in:

- `data/preprocessed_dataset.csv`

This is a structured tabular dataset and the notebook prompts the user for inputs such as:

- dataset path
- target feature name
- test size
- random seed
- cross-validation value
- feature pair for clustering
- DBSCAN parameters (`min_samples` and `eps`)

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

## Technologies Used

This project is built primarily in Python using:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- kneed

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/saberfazliahmadi/ML_Analysis_Regression_Classification_Clustering.git
cd ML_Analysis_Regression_Classification_Clustering
```

### 2) Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed
```

### 3) Open the analysis notebook

You can run the notebook in any of the following environments:

- Jupyter Notebook
- JupyterLab
- Google Colab

Open the project notebook here:

```text
notebooks/a_comprehensive_ml_analysis.ipynb
```

### 4) Run the workflow

Follow the notebook instructions and provide the required values when prompted:

- CSV file path
- target column name
- train/test split size
- random state
- cross-validation value
- clustering feature names
- DBSCAN tuning parameters

## Notes

- The notebook was originally designed with a Google Colab workflow in mind, including access to files from mounted Drive storage.
- The same logic can be adapted to a local environment by updating the dataset path and runtime setup accordingly.
- This project is especially useful as a learning example for comparing models, tuning hyperparameters, and understanding clustering behavior.

## Practical Use Cases

This kind of analysis is useful for:

- comparing model performance on the same dataset
- selecting the best model based on validation results
- interpreting prediction quality with multiple metrics
- exploring hidden patterns using clustering
- learning a full ML workflow from preprocessing to evaluation

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Saber Fazliahmadi

GitHub: https://github.com/saberfazliahmadi

---

This repository serves as a practical demonstration of machine learning analysis, covering model comparison, hyperparameter tuning, and clustering in a clear and educational workflow.
