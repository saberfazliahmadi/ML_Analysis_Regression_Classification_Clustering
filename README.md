# ML Analysis: Regression, Classification, and Clustering

This repository contains a complete machine learning analysis of a tabular dataset, focusing on model comparison, selection, and unsupervised learning. The project evaluates multiple regression and classification models, tunes hyperparameters, and applies clustering methods to uncover patterns in the data.

The work is implemented in Python and is designed as a practical, educational workflow for classical machine learning.

## Overview

The project explores a single dataset across multiple tasks:

- regression modeling and performance comparison
- classification modeling for label prediction
- hyperparameter tuning with `GridSearchCV`
- model selection based on validation metrics
- clustering analysis using K-Means++, Hierarchical Clustering, and DBSCAN

The central question is simple: which model works best for this dataset, and how does each algorithm compare under the same conditions?

## What This Project Includes

The main notebook, `notebooks/a_comprehensive_ml_analysis.ipynb`, contains a full machine learning pipeline that includes:

- dataset loading and inspection
- feature and target identification
- train/test splitting
- standardization and preprocessing
- regression model comparison
- classification model comparison
- cross-validation and hyperparameter tuning
- best-model selection
- clustering on user-selected features
- evaluation and interpretation using relevant metrics

## Dataset

The repository includes a preprocessed dataset in:

- `data/preprocessed_dataset.csv`

The dataset is a structured tabular dataset with a target column named `Class`. Based on the notebook output, it includes many encoded categorical variables and numerical launch-related attributes, such as:

- `PayloadMass`
- `Flights`
- `Block`
- `ReusedCount`
- orbit and site indicators
- other engineered features used for classification

## Machine Learning Methods Used

### Regression Models

The project evaluates classical regression approaches, including:

- Linear Regression
- Decision Tree Regressor
- other classical regression baselines used in the notebook

Common regression metrics used in the project include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Sum of Squared Errors (SSE)
- Sum of Absolute Errors (SAE)

### Classification Models

The workflow compares classical classification algorithms such as:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Classification metrics used in the notebook include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve / AUC (when applicable)

### Clustering Models

The project also explores unsupervised learning with:

- K-Means++
- Hierarchical Clustering
- DBSCAN

These clustering methods are applied to selected feature pairs to reveal natural groupings and patterns in the data.

## Typical Workflow in the Project

The analysis follows a standard end-to-end machine learning process:

1. Load and inspect the dataset.
2. Identify the target and input features.
3. Check for missing values and clean the data if needed.
4. Split data into train and test sets.
5. Scale features using `StandardScaler`.
6. Train multiple regression and classification models.
7. Evaluate performance using relevant metrics.
8. Tune hyperparameters with `GridSearchCV`.
9. Select the best-performing model.
10. Apply clustering to explore structure in the data.

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

This project is built in Python with:

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

You can run the notebook in:

- Jupyter Notebook
- JupyterLab
- Google Colab

Open:

```text
notebooks/a_comprehensive_ml_analysis.ipynb
```

### 4) Run the analysis

Follow the prompts in the notebook and provide values such as:

- dataset path
- target variable name
- train/test split ratio
- random state
- cross-validation value
- clustering feature names
- DBSCAN parameters such as `min_samples` and `eps`

## Why This Project Matters

This project is useful for learning how classical machine learning models are applied in practice:

- how models are compared on the same data
- how preprocessing and scaling affect results
- how cross-validation and tuning improve performance
- how clustering can reveal structure in feature space
- how to evaluate and interpret model quality in real workflows

## Use Cases

This project is appropriate for:

- machine learning learning and demonstration
- model comparison on a real tabular dataset
- experimentation with regression and classification methods
- exploring clustering behavior in structured data
- understanding a full ML pipeline from preprocessing to evaluation

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Saber Fazliahmadi

GitHub: https://github.com/saberfazliahmadi

---

This repository demonstrates a practical machine learning workflow focused on regression, classification, and clustering, highlighting model comparison, tuning, and interpretation in a clear and educational way.
