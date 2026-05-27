# Iris Decision Tree Classification — End-to-End ML Project

A beginner-friendly, modular machine learning project that classifies Iris flower species using **Decision Tree Classifier** on the built-in scikit-learn Iris dataset.

## Project Overview

This project demonstrates a complete ML workflow:

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_ingestion.py` | Load Iris dataset, convert to DataFrame |
| 2 | `data_cleaning.py` | Handle missing values, remove duplicates |
| 3 | `outlier_handling.py` | Detect & cap outliers using IQR |
| 4 | `preprocessing.py` | Train-test split with stratification |
| 5 | `model_training.py` | Decision Tree with hyperparameter tuning (max_depth, min_samples_split) |
| 6 | `evaluation.py` | Accuracy, precision, recall, F1 |
| 7 | `prediction.py` | Load artifacts and predict |
| 8 | `app/app.py` | Streamlit web interface |

## Project Structure

```
DecisionTreeClassifier/
│
├── artifacts/              # Generated models & data (after running main.py)
│   ├── processed_data.csv
│   ├── model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── notebooks/
│   └── experiments.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── outlier_handling.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── prediction.py
│   ├── utils.py
│   └── logger.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
├── main.py
└── logs.log          # Generated after running main.py
```
└── .gitignore
```

## Dataset

- **Source:** `sklearn.datasets.load_iris()`
- **Features:** sepal length, sepal width, petal length, petal width (cm)
- **Target:** `target` (0 = Setosa, 1 = Versicolor, 2 = Virginica)

## Setup & Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This will:
- Load and process the Iris dataset
- Train a KNN model (tuning K from 1 to 20)
- Save artifacts to `artifacts/`
- Print evaluation metrics
- Run a sample prediction

### 5. Launch the Streamlit app

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`) and enter flower measurements to get a prediction.

## Key Concepts

### Outlier Handling (IQR Method)

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Outliers are **capped** (not removed) to preserve row count.

### KNN Hyperparameter Tuning

The model tries `K = 1` through `K = 20` and selects the value with the highest test accuracy.

### Feature Scaling

KNN is distance-based, so features are scaled with `StandardScaler` (zero mean, unit variance).

## Artifacts

| File | Description |
|------|-------------|
| `model.pkl` | Trained KNN classifier |
| `scaler.pkl` | Fitted StandardScaler |
| `encoder.pkl` | LabelEncoder for class names |
| `processed_data.csv` | Cleaned & processed dataset |

## Logs

All pipeline steps are logged to `logs.log` in the project root.

## Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn
- joblib
- streamlit
- matplotlib, seaborn

## License

Educational use — free to modify and learn from.
