# ML Analysis: Regression, Classification, and Clustering

This repository contains a comprehensive machine learning workflow for analyzing a tabular dataset using both supervised and unsupervised learning techniques. The project compares regression and classification models, performs hyperparameter tuning, and applies clustering methods to identify patterns in the data.

The notebook is designed as an educational and practical example of how to build a complete ML pipeline in Python, from preprocessing and model comparison to evaluation and interpretation.

## Project Overview

The project answers a common machine learning question:

- Which model performs best on this dataset?
- How do different regression models compare?
- How do classification models compare under the same conditions?
- Can clustering reveal meaningful structure in the data?

To address these questions, the workflow includes:

- loading and inspecting the dataset
- splitting data into train/test sets
- standardizing feature values for consistent model comparison
- comparing regression models using error metrics
- comparing classification models using accuracy and classification metrics
- tuning hyperparameters with GridSearchCV
- selecting the best-performing model
- applying K-Means++, Hierarchical Clustering, and DBSCAN to selected features

## Dataset

The repository includes a preprocessed dataset at:

- `data/preprocessed_dataset.csv`

The notebook and dataset are structured around a rocket/launch-style tabular dataset, with a target variable named `Class` and many engineered or encoded feature columns. Based on the notebook outputs, the dataset contains:

- 90 rows
- 89 columns
- a binary classification target (`Class`)
- multiple categorical/one-hot encoded features
- numerical features such as `PayloadMass`, `Flights`, `Block`, and others

## Notebook

The main project notebook is:

- `notebooks/a_comprehensive_ml_analysis.ipynb`

It contains the full analysis pipeline and includes interactive input prompts for:

- dataset path
- target variable name
- train/test split ratio
- random state
- cross-validation value
- clustering feature selection
- DBSCAN parameters (`min_samples` and `eps`)

## Included Methods

### Regression analysis

The workflow evaluates common regression models, including:

- Linear Regression
- Decision Tree Regressor
- other classical regression baselines used in the notebook

Regression metrics used in the project include:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Sum of Squared Errors (SSE)
- Sum of Absolute Errors (SAE)

### Classification analysis

The notebook compares several classical classification models, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Classification evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve / AUC (where applicable)

### Clustering analysis

The project also explores unsupervised learning with:

- K-Means++
- Hierarchical Clustering
- DBSCAN

These clustering methods are applied to selected feature pairs to identify groups and patterns in the data.

## Machine Learning Workflow

The project follows a typical end-to-end analysis process:

1. Load the dataset.
2. Inspect and validate the data.
3. Check for missing values and clean the dataset.
4. Split into train/test data.
5. Standardize features with `StandardScaler`.
6. Train multiple regression and classification models.
7. Evaluate models with relevant metrics.
8. Use `GridSearchCV` for hyperparameter tuning.
9. Select the best-performing model.
10. Apply clustering to compare structural patterns in the feature space.

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

The project is implemented in Python and relies on the following libraries:

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

### 4) Run the workflow

Follow the notebook prompts and provide the required values:

- dataset path
- target variable name
- train/test split ratio
- random state
- cross-validation value
- clustering feature names
- DBSCAN parameters (`min_samples` and `eps`)

## Notes

- The notebook was originally designed for a Google Colab workflow, including mounted Drive access.
- The workflow is easy to adapt for a local environment by changing the dataset path and file access logic.
- This project functions as a hands-on educational example for model comparison, tuning, and clustering in machine learning.

## Use Cases

This project is useful for:

- comparing model performance on the same dataset
- learning how to tune models with cross-validation
- understanding regression and classification evaluation metrics
- exploring clusters in a structured dataset
- building an end-to-end ML workflow for experimentation and learning

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Saber Fazliahmadi

GitHub: https://github.com/saberfazliahmadi

---

This repository is a practical demonstration of a full machine learning analysis pipeline, covering data preparation, model comparison, hyperparameter tuning, and clustering on a real dataset.
