# ML Analysis: Regression, Classification, and Clustering

A complete machine learning project that explores a tabular dataset through a full model comparison workflow. This repository demonstrates how to preprocess data, train multiple machine learning models, evaluate performance, tune hyperparameters, and apply clustering techniques to understand structure in the data.

The project is implemented primarily in Python and is designed as a practical, educational workflow for machine learning experimentation.

## Overview

This project addresses a common data science problem: given a dataset, which model performs best, and how do different algorithms compare under the same conditions?

The workflow includes:

- dataset loading and inspection
- target and feature separation
- train/test splitting
- feature scaling and preprocessing
- regression model comparison
- classification model comparison
- hyperparameter tuning with `GridSearchCV`
- best-model selection based on validation performance
- clustering using unsupervised methods
- metric interpretation and model evaluation

## What the Project Does

This repository contains a single end-to-end notebook that compares classical machine learning models on the same dataset and then applies clustering methods to selected features.

### Regression analysis

The workflow evaluates common regression models, including:

- Linear Regression
- Decision Tree Regressor
- additional classical regression baselines used in the notebook

Regression metrics used in the project include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Sum of Squared Errors (SSE)
- Sum of Absolute Errors (SAE)

### Classification analysis

The notebook compares a range of classic classification algorithms, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Classification metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve and AUC (when applicable)

### Clustering analysis

The project also explores unsupervised learning with:

- K-Means++
- Hierarchical Clustering
- DBSCAN

These clustering methods are applied to selected feature pairs to reveal groupings and relationships in the dataset.

## Dataset

The repository includes a preprocessed dataset:

- `data/preprocessed_dataset.csv`

From the notebook output, the dataset appears to be a structured tabular dataset with many one-hot encoded and numerical features, where the target variable is `Class`.

The dataset contains a combination of:

- binary and categorical indicator variables
- numerical variables such as `PayloadMass`, `Flights`, `Block`, and related launch-related attributes
- a classification target used for supervised learning

The notebook is designed to let the user provide:

- the path to the dataset
- the target feature name
- the train/test split ratio
- the random seed
- the number of cross-validation folds
- the feature pair used for clustering
- DBSCAN values such as `min_samples` and `eps`

## Notebook

The main analysis notebook is:

- `notebooks/a_comprehensive_ml_analysis.ipynb`

This notebook contains the full workflow and is designed for interactive execution in either Jupyter Notebook or Google Colab.

## Typical Workflow in the Project

The project follows a standard machine learning pipeline:

1. Load the dataset.
2. Check and inspect data shape and columns.
3. Identify the target variable.
4. Remove or handle missing values, if needed.
5. Split the data into train and test sets.
6. Scale features with `StandardScaler`.
7. Train multiple regression and classification models.
8. Compare results with evaluation metrics.
9. Tune hyperparameters using `GridSearchCV`.
10. Choose the best-performing model.
11. Apply clustering to selected feature pairs.
12. Interpret patterns and model performance.

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

This project is implemented in Python using:

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

### 3) Open the notebook

Run the project in:

- Jupyter Notebook
- JupyterLab
- Google Colab

Open the notebook file:

```text
notebooks/a_comprehensive_ml_analysis.ipynb
```

### 4) Run the analysis

Follow the notebook prompts to provide values such as:

- dataset path
- target variable name
- test size for train/test split
- random state
- cross-validation parameter
- selected clustering features
- DBSCAN configuration (`min_samples`, `eps`)

## Project Highlights

- end-to-end machine learning workflow in a single notebook
- comparison of multiple regression and classification models
- systematic model selection with cross-validation and tuning
- practical use of clustering to explore patterns in the data
- suitable for learning, experimentation, and demonstration purposes

## Notes

- The notebook was originally designed with a Google Colab workflow in mind, including Drive-mounted file access.
- It can be adapted to local environments by updating the dataset path and file access logic.
- This is a strong educational project for understanding how classical ML models are compared and tuned in practice.

## Use Cases

This project is useful for:

- comparing model performance on the same dataset
- learning best practices in preprocessing and evaluation
- understanding how hyperparameter tuning improves model performance
- applying clustering to structured tabular data
- demonstrating a practical ML workflow from start to finish

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Saber Fazliahmadi

GitHub: https://github.com/saberfazliahmadi

---

This repository is a practical machine learning analysis project focused on regression, classification, and clustering, with a clear emphasis on comparison, tuning, and interpretation of model performance.
