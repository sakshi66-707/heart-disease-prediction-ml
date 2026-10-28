# Heart Disease Prediction Using Machine Learning

## Project Overview

This project explores the prediction of heart disease using supervised machine learning algorithms. It is an improved version of my undergraduate machine learning project from 2022, revisited with a focus on data quality, preprocessing, model evaluation, and reproducibility.

The objective is to compare classification algorithms and investigate how preprocessing and hyperparameter tuning affect predictive performance.

**Note:** This is an educational machine learning project, not a medical diagnostic tool.

## Dataset

**Source:** [Heart Disease Dataset — Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

The dataset contains patient-related attributes such as age, cholesterol, resting blood pressure, maximum heart rate, and other cardiovascular indicators.

- Original dataset: 1,025 rows and 14 columns
- Duplicate rows removed: 723
- Unique records after cleaning: 302
- Target variable: `target`
- Classification task: Binary classification

Duplicate records were removed before splitting the data to reduce the risk of identical records appearing in both training and testing.

## Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas and NumPy
- Matplotlib
- Scikit-learn

## Machine Learning Models

The following algorithms were evaluated:

1. Logistic Regression
2. Gaussian Naive Bayes
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. Improved KNN with feature-specific preprocessing and hyperparameter tuning

## Methodology

The project follows these steps:

1. Load and explore the dataset.
2. Identify and remove duplicate records.
3. Split the cleaned data into stratified training and testing sets.
4. Apply preprocessing within machine learning pipelines.
5. Train and evaluate baseline classification models.
6. Improve KNN using numerical feature scaling, categorical feature encoding, and GridSearchCV.
7. Evaluate the improved KNN approach using nested five-fold cross-validation.

## Model Performance

Results from the 80/20 train-test split:

| Model                  | Test Accuracy |
| ---------------------- | ------------- |
| Logistic Regression    | 80.33%        |
| Gaussian Naive Bayes   | 78.69%        |
| KNN (K=12)             | 80.33%        |
| Random Forest          | 77.05%        |
| **Improved KNN (K=5)** | **83.61%**    |

Improved KNN achieved the highest accuracy on this particular test split.

### Nested Cross-Validation

To obtain a more robust performance estimate, nested five-fold cross-validation was performed for the improved KNN approach.

- Mean accuracy: **80.78%**
- Standard deviation: **5.37 percentage points**

The difference between test accuracy and cross-validation accuracy illustrates how model performance can vary across data splits.

## Improvements Over the Original Project

Compared with the undergraduate version, this project introduces:

- Duplicate detection and removal before model training
- Consistent train-test splitting for baseline comparisons
- Preprocessing pipelines to reduce data leakage
- Separate treatment of numerical and categorical features
- Hyperparameter tuning with GridSearchCV
- Evaluation using accuracy, precision, recall, and F1-score
- Nested cross-validation for improved KNN

These improvements strengthen the evaluation methodology, although they do not guarantee higher predictive accuracy.

## Key Findings

- Data quality and leakage prevention are important for reliable model evaluation.
- Improved KNN achieved the highest observed accuracy on the selected test split.
- Random Forest did not outperform KNN in the baseline experiment.
- Cross-validation revealed variation in performance across different subsets of the data.
- Higher accuracy on one test split does not necessarily indicate better generalization.

## Limitations

- The cleaned dataset contains only 302 unique records.
- The test set contains 61 records, making its accuracy estimate sensitive to individual predictions.
- Nested cross-validation was performed for improved KNN, not for every algorithm.
- The dataset is intended for educational experimentation and does not establish clinical validity.

## How to Run

1. Download or open `Heart_Disease_Prediction_GitHub.ipynb` in Google Colab.
2. Download `archive.zip` from the Kaggle dataset linked above.
3. Run the notebook's upload cell and select `archive.zip`.
4. Run the remaining cells in order.

The notebook contains the preprocessing, model training, evaluation, and cross-validation workflow.

## Future Improvements

Potential extensions include evaluating additional algorithms with nested cross-validation, studying feature importance, and validating the approach on an independent dataset.

---

**Project Type:** Machine Learning | Binary Classification | Python

**Project Background:** Undergraduate machine learning project (2022), revisited and improved (2026).
