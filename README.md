<div align="center">

# 🩺 Diabetes Risk Prediction
### Machine Learning · Health Awareness · Streamlit

A machine learning web application that predicts a **Low, Moderate, or High dataset risk category** from demographic, lifestyle, and health-related inputs.

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Joblib-4B8BBE?style=for-the-badge&logo=python&logoColor=white" alt="Joblib">
</p>

<p>
  <img src="https://img.shields.io/badge/Task-Multiclass%20Classification-477F79?style=flat-square" alt="Multiclass classification">
  <img src="https://img.shields.io/badge/Status-Educational%20Project-6C757D?style=flat-square" alt="Educational project">
  <img src="https://img.shields.io/badge/SDG-3%20Good%20Health%20%26%20Well--Being-E5243B?style=flat-square" alt="SDG 3">
</p>

</div>

> **Important:** This project is for educational and awareness purposes only. Its predictions reproduce categories learned from the supplied dataset; they are not a medical diagnosis, a clinically validated risk assessment, or a substitute for professional medical advice.

---

## 📌 Project Overview

This project demonstrates an end-to-end machine learning workflow, from preparing a tabular dataset to training and evaluating a classifier and integrating the saved pipeline into a Streamlit web application.

Users enter 17 demographic, lifestyle, and health-related features. The app passes the submitted values to a saved scikit-learn pipeline and displays one of three predicted dataset categories: **Low**, **Moderate**, or **High**.

The project is aligned with **United Nations Sustainable Development Goal 3 (Good Health and Well-being)** as an educational exploration of data-driven health awareness. This alignment does not imply clinical validation.

## 🌐 Live Demo

<p align="center">
  <a href="https://diabetes-risk-prediction01.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
</p>

Try the interactive Diabetes Risk Prediction application built with Python, Scikit-learn, and Streamlit.

> **Disclaimer:** This project is for educational purposes only. Predictions are not a medical diagnosis or a clinically validated risk assessment.


## ✨ Features

- 🧠 **Multi-class prediction** — predicts Low, Moderate, or High dataset categories.
- 🧹 **Missing-value handling** — imputes numeric and categorical values.
- 🔠 **Categorical encoding** — uses one-hot encoding for categorical features.
- 📏 **Numeric scaling** — standardises numeric features.
- ⚖️ **Class imbalance handling** — uses balanced class weights during training.
- 🔬 **Model evaluation** — reports accuracy, precision, recall, F1-score, and a confusion matrix.
- 💾 **Reusable ML pipeline** — saves preprocessing and model steps together.
- 🖥️ **Interactive interface** — accepts inputs through a Streamlit form and displays the submitted values.
- 🔁 **Reproducible split** — uses a stratified 80/20 train-test split with a fixed random seed.

## 🗂️ Dataset

The project uses a Kaggle dataset containing **15,000 records and 19 columns**.

| Item | Details |
|---|---|
| Target column | `diabetes_risk` |
| Identifier excluded from training | `patient_id` |
| Input features | 17 |
| Target classes | Low, Moderate, High |
| Records | 15,000 |

### Target distribution

| Category | Records | Share |
|---|---:|---:|
| Low | 9,000 | 60% |
| Moderate | 3,750 | 25% |
| High | 2,250 | 15% |
| **Total** | **15,000** | **100%** |

The target classes are imbalanced. The model uses `class_weight="balanced"` to adjust class weights during training, but this does not guarantee equal performance for every class. Per-class evaluation metrics remain important.

### Input features

| Feature | Data type |
|---|---|
| `age` | Numeric |
| `gender` | Categorical |
| `city` | Categorical |
| `bmi` | Numeric |
| `family_history_diabetes` | Categorical |
| `physical_activity_level` | Categorical |
| `diet_type` | Categorical |
| `smoking_status` | Categorical |
| `alcohol_consumption` | Categorical |
| `hours_sleep_per_night` | Numeric |
| `stress_level` | Numeric |
| `fasting_blood_sugar` | Numeric |
| `hba1c_level` | Numeric |
| `blood_pressure_systolic` | Numeric |
| `blood_pressure_diastolic` | Numeric |
| `waist_circumference_cm` | Numeric |
| `income_bracket` | Categorical |

Missing values were observed in `smoking_status` (472 records), `alcohol_consumption` (3,788 records), and `income_bracket` (463 records). The training pipeline handles missing values during preprocessing.

**Dataset note:** Check the original Kaggle page for the dataset source, licence, collection method, definitions, and representativeness before using or redistributing the data.

## ⚙️ Machine Learning Workflow

```text
Kaggle CSV
    ↓
Load data and remove patient_id
    ↓
Separate input features (X) and target (y)
    ↓
Stratified train-test split (80/20)
    ↓
Preprocess numeric and categorical features
    ↓
Train Logistic Regression
    ↓
Evaluate on held-out test data
    ↓
Save fitted pipeline with Joblib
    ↓
Load pipeline in Streamlit
    ↓
User inputs → Predicted dataset category
```

### 1. Data preparation

The `patient_id` identifier is excluded from model training. The remaining 17 columns are used as input features (`X`), while `diabetes_risk` is the target (`y`).

### 2. Train-test split

The data is divided using `test_size=0.20`, `random_state=42`, and `stratify=y`.

- **Training set:** 12,000 records
- **Test set:** 3,000 records

Stratification keeps class proportions approximately consistent across the two subsets.

### 3. Preprocessing

A scikit-learn `ColumnTransformer` applies different preprocessing steps to numeric and categorical features:

| Feature group | Preprocessing |
|---|---|
| Numeric | Median imputation → `StandardScaler` |
| Categorical | Most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")` |

The preprocessing steps and classifier are combined in one scikit-learn `Pipeline`. The pipeline is fitted on training data, helping prevent preprocessing leakage from the held-out test set.

### 4. Model

The project uses **Logistic Regression** as a baseline multi-class classifier.

- `max_iter=1000` allows more optimisation iterations.
- `class_weight="balanced"` adjusts weights based on class frequencies.

The model's suitability should be judged using measured test performance, not assumed in advance.

### 5. Evaluation

The training script reports:

- **Accuracy:** proportion of test records predicted correctly.
- **Precision:** how many predictions for a class are correct.
- **Recall:** how many actual records of a class are identified.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** actual versus predicted class counts.

No performance score is included here because it should be taken from the actual output of `train_model.py`. For an imbalanced dataset, review per-class precision, recall, and F1-score alongside accuracy.

## 🖥️ Application Workflow

1. Streamlit loads `model/diabetes_risk_model.pkl`.
2. The user enters values for the 17 input features.
3. On form submission, the app creates a one-row pandas DataFrame with the feature names expected by the model.
4. The saved pipeline applies its fitted preprocessing transformations.
5. The classifier returns a predicted dataset category.
6. The app displays the prediction and allows the submitted values to be reviewed.

The app loads the saved pipeline rather than retraining the model for each prediction.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| <img src="https://cdn.simpleicons.org/python/3776AB" width="20" alt="Python"> **Python** | Application and model code |
| <img src="https://cdn.simpleicons.org/pandas/150458" width="20" alt="Pandas"> **pandas** | CSV loading and tabular data handling |
| <img src="https://cdn.simpleicons.org/scikitlearn/F7931E" width="20" alt="scikit-learn"> **scikit-learn** | Preprocessing, classification, splitting, and evaluation |
| <img src="https://cdn.simpleicons.org/streamlit/FF4B4B" width="20" alt="Streamlit"> **Streamlit** | Interactive web interface |
| <img src="https://cdn.simpleicons.org/python/4B8BBE" width="20" alt="Joblib"> **Joblib** | Saving and loading the fitted pipeline |

## 📁 Project Structure

```text
Diabetes-Risk-Prediction/
├── data/
│   └── diabetes.csv
├── model/
│   └── diabetes_risk_model.pkl
├── app.py
├── check_dataset.py
├── preprocess_data.py
├── README.md
├── requirements.txt
├── test_model.py
├── train_model.py
└── .gitignore   
```

## 🚀 Run Locally

### Prerequisites

- Python installed on your system
- Git (optional, if cloning the repository)
- The dataset at `data/diabetes.csv`

### 1. Clone the repository

Replace the placeholder with your actual repository URL:

```bash
git clone YOUR_REPOSITORY_URL
cd Diabetes-Risk-Prediction
```

If you already have the project folder, open that folder in your terminal instead.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Make sure `requirements.txt` contains the libraries used by the project, with compatible versions:

```text
streamlit
pandas
scikit-learn
joblib
```

Then run:

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

This trains the pipeline, prints evaluation metrics, and saves `model/diabetes_risk_model.pkl`.

### 5. Test the saved pipeline

```bash
python test_model.py
```

This checks that the saved pipeline loads and produces predictions on the same reproducible test split. It is a software check, not independent external validation.

### 6. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, enter the required values, and click **Predict Risk Category**.

## 🧪 Testing

- **Model evaluation:** inspect accuracy, per-class precision/recall/F1, and the confusion matrix on the held-out test set.
- **Saved-model test:** confirm the pipeline loads and predicts for all 3,000 test records.
- **UI test:** submit different input combinations and verify the displayed submitted values and prediction.
- **Input handling:** check that supported categorical values and missing-value preprocessing work as expected.

A prediction can remain in the same category after an input changes. Confirm the submitted values first, then assess model behaviour using evaluation metrics rather than individual examples alone.

## ⚠️ Limitations

- The model learns the target labels present in the supplied dataset; clinical validity has not been established.
- The dataset's provenance, licence, collection method, and representativeness require verification.
- Target classes are imbalanced.
- A held-out split from the same dataset is not independent external validation.
- The predicted category is not a calibrated probability of developing diabetes.
- The output must not be used to make diagnosis or treatment decisions.

## 🔮 Future Improvements

- Compare additional classifiers using a consistent evaluation protocol.
- Use cross-validation on training data for model selection.
- Add exploratory visualisations and class-wise performance charts.
- Pin compatible dependency versions for reproducible setup and deployment.
- Add automated tests for input validation, model loading, and prediction flow.
- Evaluate on suitable independent data before making real-world claims.

## 👤 About the Author

**Faisal Khan**  


---

<div align="center">

**Built for learning and health awareness · SDG 3: Good Health and Well-being**

</div>
