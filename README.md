# NYC Taxi Fare Prediction System

## Project Overview

This project implements an end-to-end machine learning system for predicting New York City taxi fares. It utilizes a regression pipeline built with Scikit-learn to estimate fare amounts based on trip details such as distance, passenger count, pickup/dropoff locations, and time of day.

**Problem Statement**: Accurately predicting taxi fares is critical for rider transparency and driver earnings estimation. This system solves a regression problem where the target variable is the total `fare_amount`.

**Key Features**:
*   Automated data ingestion and cleaning pipeline.
*   Feature engineering extracting temporal and spatial insights.
*   Stratified data splitting to handle fare distribution imbalances.
*   Model training with baseline, linear, and ensemble methods.
*   FastAPI backend for real-time inference.
*   Modern, responsive frontend for user interaction.

---

## Dataset Information

**Source**: New York City Taxi and Limousine Commission (TLC) Trip Record Data (Yellow Taxi).
**File Structure**: The system expects a raw CSV file at `data/raw/raw.csv`.

### Features Used

The model uses the following input features:

*   **Numerical Features**:
    *   `trip_distance`: Distance of the trip in miles.
    *   `passenger_count`: Number of passengers.
    *   `tpep_pickup_datetime`: Used to derive temporal features (Hour, Day of Week, Month).
*   **Categorical Features**:
    *   `PULocationID`: TLC Taxi Zone in which the taximeter was engaged.
    *   `DOLocationID`: TLC Taxi Zone in which the taximeter was disengaged.
    *   `VendorID`: Code indicating the LPEP provider that provided the record.
    *   `RatecodeID`: The final rate code in effect at the end of the trip.

### Target Variable
*   **`fare_amount`**: The time-and-distance fare calculated by the meter.

### Cleaning & Filtering Rules
Data is cleaned in `src/data_ingestion.py` based on the following assumptions:
*   **Fare Constraints**: Rows with `fare_amount` <= 0 or >= $500 are removed to exclude outliers/errors.
*   **Passenger Constraints**: Trips must have between 1 and 6 passengers.
*   **Missing Values**: Rows with missing target values are dropped.

---

## Project Architecture

The project follows a modular structure for reproducibility and maintainability.

```
regressionProject/
├── api/                   # Backend application
│   └── main.py            # FastAPI service exposing /predict endpoint
├── data/                  # Data storage (gitignored except placeholders)
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned and split data (train/val/test)
├── frontend/              # User Interface
│   ├── index.html         # Main prediction dashboard
│   ├── styles.css         # Styling
│   └── script.js          # API integration logic
├── models/                # Serialized model artifacts
│   └── model_v1.joblib    # Trained best-performing model
├── src/                   # ML Pipeline Source Code
│   ├── data_ingestion.py  # Loads and cleans raw data
│   ├── feature_engineering.py # Transformations (time extraction, log distance)
│   ├── preprocessing.py   # Scikit-learn pipelines (imputation, scaling, encoding)
│   ├── split.py           # Stratified train/val/test splitting
│   └── train.py           # Model training and evaluation script
├── environment.yml        # Conda environment specification
└── README.md              # Project documentation
```

### Data Flow
1.  **Ingestion (`data_ingestion.py`)**: Raw CSV → Cleaned DataFrame.
2.  **Splitting (`split.py`)**: Cleaned Data → Stratified Split (Train, Val, Test).
3.  **Training (`train.py`)**: Train Data → Preprocessing Pipeline → Model → Evaluation.
4.  **Inference**: User Input → Frontend → API → Model → Prediction.

---

## Technical Stack

*   **Language**: Python 3.9+
*   **Libraries**:
    *   **Data Manipulation**: Pandas, NumPy
    *   **Machine Learning**: Scikit-learn
    *   **API Framework**: FastAPI, Uvicorn
    *   **Serialization**: Joblib
*   **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
*   **Environment Management**: Conda / Pip

---

## Data Processing Pipeline

### Feature Engineering (`src/feature_engineering.py`)
Custom transformers create new predictive signals:
*   **`pickup_hour`, `pickup_dayofweek`, `pickup_month`**: Extracted from datetime to capture traffic patterns.
*   **`is_weekend`**: Boolean flag for weekend trips.
*   **`log_distance`**: Log-transformation (`log1p`) of trip distance to normalize right-skewed distributions.
*   **`is_zero_distance`**: Flag for trips with 0 distance (potential errors or cancelled rides).
*   **`is_single_passenger`**: Boolean flag for solo riders.

### Preprocessing (`src/preprocessing.py`)
A `ColumnTransformer` handles data preparation:
*   **Numeric Pipeline**: Median Imputation → Standard Scaling.
*   **Categorical Pipeline**: Most Frequent Imputation → One-Hot Encoding (handling unknown categories and infrequent levels).

---

## Model Development

### Models Evaluated (`src/train.py`)
1.  **Baseline Model**: `DummyRegressor` (predicts median fare). Used to establish a performance floor.
2.  **Linear Regression**: Simple interpretable model to capture linear relationships between distance and fare.
3.  **Random Forest Regressor**: Ensemble method to capture non-linear interactions. Configured with `n_estimators=40` and `max_depth=12`.

### Evaluation
*   **Metric**: Root Mean Squared Error (RMSE).
*   **Validation Strategy**: Models are trained on the training set and evaluated on a hold-out validation set. The model with the lowest validation RMSE is serialized.

### Reproducibility: Stratified Splitting
To ensure the model performs well across all price points, `src/split.py` uses **stratified splitting**. Fares are bucketed into ranges (0-5, 5-10, 10-20, etc.), ensuring train/val/test sets have identical target distributions.

---

## User Workflow

### 1. Environment Setup
Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate regression_env
# OR using pip
pip install -r requirements.txt
```

### 2. Data Preparation
Place your raw data file at `data/raw/raw.csv`. Then run the pipeline steps:

```bash
# Clean raw data
python -m src.data_ingestion

# Create train/val/test splits
python -m src.split
```

### 3. Model Training
Train the models and save the best one:
```bash
python -m src.train
```
*Output*: Normalized RMSE scores for tested models and a saved artifact at `models/model_v1.joblib`.

### 4. Running the Application
Start the backend API and frontend:

**Backend**:
```bash
uvicorn api.main:app --reload
```
*API will be available at http://127.0.0.1:8000*

**Frontend**:
Serve the `frontend` directory using a simple HTTP server:
```bash
python -m http.server 5500 --directory frontend
```
Open a browser to `http://localhost:5500`.

---

## Assumptions and Design Decisions

*   **High Cardinality Categoricals**: Feature hashing or specific encoding strategies might be needed for `LocationID` in production with larger datasets; currently, `OneHotEncoder` with `min_frequency` handles rare zones effectively.
*   **Outlier Removal**: 500 USD cap was chosen based on inspection of typical NYC taxi fares to avoid skewing the MSE loss function.
*   **Model Selection**: Random Forest was chosen for its ability to handle non-linearities and interactions without extensive feature scaling requirements, though Linear Regression provides a strong baseline.
*   **Stratification**: Essential because high-fare trips are rare but critical to predict accurately. Random splitting would likely result in poor performance on expensive trips.

## Future Improvements

*   **Model**: Experiment with XGBoost or LightGBM for better performance and speed.
*   **Features**: Integrate external weather or traffic data API.
*   **Deploy**: Dockerize the application for containerized deployment (Dockerfile included).
*   **Monitoring**: Add drift detection for input data distributions (e.g., using Evidently AI).
