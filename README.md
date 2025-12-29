# Cinema Audience Forecasting – End‑to‑End ML Pipeline

This repository contains a complete end‑to‑end machine learning pipeline for forecasting **daily audience counts** for movie theaters using the **Cinema Audience Forecasting** Kaggle challenge data.[file:26]  
The core goal is to predict how many people will visit a theater on a given day, using the `audience_count` column from the `booknow_visits` table as the target variable.[file:26]

---

## Project Goal

The main objective of this project is:

- To build a **tabular regression model** that predicts `audience_count` for each `(book_theater_id, show_date)` combination.[file:26]
- To design a pipeline that is:
  - Data‑aware (uses theater metadata, bookings, and calendar information),
  - Interpretable (feature engineering is transparent),
  - And competitive enough to be used for Kaggle submissions.[file:26]

---

## Dataset Description

All datasets come from the Kaggle **Cinema Audience Forecasting** competition and are loaded from the `/kaggle/input/Cinema_Audience_Forecasting_challenge/` directory.[file:26]

The notebook loads and uses the following CSV files:

- `cinePOS_theaters.csv`  
  - Theater metadata from the cinePOS system (`cine_theater_id`, `theater_type`, `theater_area`, `latitude`, `longitude`).[file:26]
- `booknow_theaters.csv`  
  - Theater metadata from the booknow system (`book_theater_id`, `theater_type`, `theater_area`, `latitude`, `longitude`).[file:26]
- `movie_theater_id_relation.csv`  
  - Mapping table linking `book_theater_id` and `cine_theater_id` so information across both systems can be combined.[file:26]
- `cinePOS_booking.csv`  
  - Ticket booking logs from cinePOS (`cine_theater_id`, `show_datetime`, `booking_datetime`, `tickets_sold`).[file:26]
- `booknow_booking.csv`  
  - Ticket booking logs from booknow (`book_theater_id`, `show_datetime`, `booking_datetime`, `tickets_booked`).[file:26]
- `booknow_visits.csv`  
  - **Main training target**: historical visits per theater per date (`book_theater_id`, `show_date`, `audience_count`).[file:26]
- `date_info.csv`  
  - Calendar information for each date (`show_date`, `day_of_week`).[file:26]
- `sample_submission.csv`  
  - Submission format for the Kaggle competition (`ID`, `audience_count`).[file:26]

---

## High‑Level Pipeline

The notebook follows a clear, step‑by‑step structure:

1. **Library imports and dataset loading**  
   - Imports core libraries (NumPy, pandas, Matplotlib) and scikit‑learn components for preprocessing, modeling, and evaluation.[file:26]
   - Loads all CSVs listed above and prints “All datasets loaded successfullyyy!!!!!” to confirm success.[file:26]

2. **Exploratory Data Analysis (EDA)**  
   - For each table, the notebook inspects:
     - Shape (rows, columns),
     - Column names and dtypes,
     - First few rows,
     - Missing values per column,
     - Basic descriptive statistics.[file:26]
   - This step provides a clear understanding of what each table contributes (theater metadata, bookings, visits, calendar).[file:26]

3. **Table merging and master dataset construction**  
   The notebook builds a **master modeling table** by carefully merging multiple sources:

   - Join `booknow_visits` (base target table) with `date_info` to add `day_of_week`.[file:26]
   - Merge with `booknow_theaters` to enrich each theater with `theater_type`, `theater_area`, `latitude`, `longitude`.[file:26]
   - Use `movie_theater_id_relation` to link booknow theaters with cinePOS theaters.[file:26]
   - Aggregate `booknow_booking` and `cinePOS_booking` to get daily booking statistics per theater, such as:
     - `booknow_tickets`, `booknow_count`,
     - `cine_tickets`, `cine_count`,
     - `total_tickets`, `total_bookings`.[file:26]
   - Combine these aggregates back into the main table.[file:26]

   The result is a feature‑rich dataset where each row corresponds to a single `(book_theater_id, show_date)` with both historical audience info and booking behavior around that date.[file:26]

4. **Feature engineering**

   The notebook creates several important engineered features to improve predictive power:

   - **Date features** from `show_date`:
     - `year`, `month`, `day`, `day_of_year`, `week_of_year`, `quarter`.[file:26]
   - **Categorical information**:
     - `day_of_week`,
     - `theater_type`,
     - `theater_area`.[file:26]
   - **Spatial features**:
     - `latitude`, `longitude` from theater tables.[file:26]
   - **Booking‑based aggregates**:
     - `booknow_tickets`, `booknow_count`, `cine_tickets`, `cine_count`, and their combined `total_tickets` and `total_bookings` per theater‑date.[file:26]
   - **Theater‑level audience statistics**:
     - `theater_avg_audience`,
     - `theater_median_audience`,
     - `theater_std_audience`,
     - `theater_max_audience`,
     - `theater_min_audience` computed from historical `audience_count` per theater.[file:26]

   The intent is to give the model both **local booking context** and **long‑term theater behaviour** so it can generalize better beyond raw counts.

5. **Train/test splitting and preprocessing**

   - The target variable is `audience_count` from the enriched visits table.[file:26]
   - Feature set includes:
     - Date features,
     - Calendar and categorical encodings,
     - Theater location/type/area,
     - Booking aggregates,
     - Theater‑level audience statistics.[file:26]
   - The dataset is split into training and validation sets using `train_test_split`.[file:26]
   - `StandardScaler` is used where appropriate to standardize numerical features for certain models.[file:26]
   - Categorical columns are label‑encoded where needed via `LabelEncoder`.[file:26]

---

## Models Implemented

The notebook experiments with multiple regression models to understand what works best for this forecasting problem.[file:26]

### Baseline

- **DummyRegressor**
  - Acts as a naive baseline that predicts a simple statistic (e.g., mean of `audience_count`).[file:26]
  - Helps quantify how much value complex models add.

### Linear Models

- **LinearRegression**
  - Ordinary least squares regression to model a linear relationship between engineered features and `audience_count`.[file:26]
- **Ridge Regression**
  - Linear regression with L2 regularization to handle multicollinearity and reduce variance.[file:26]
- **Lasso Regression**
  - Linear regression with L1 regularization for potential feature selection effects.[file:26]
- **ElasticNet**
  - Combines L1 and L2 regularization for a balance between feature selection and stability.[file:26]

### Tree‑Based Models

- **DecisionTreeRegressor**
  - Single decision tree modeling nonlinear feature interactions.[file:26]
  - Provides interpretability and helps understand important splits.

### Ensemble Models

- **RandomForestRegressor**
  - Bagging ensemble of many decision trees with feature subsampling.[file:26]
  - Good at capturing non‑linear relationships and reducing variance relative to a single tree.

- **GradientBoostingRegressor**
  - Boosting ensemble that builds trees sequentially, focusing on residual errors of previous trees.[file:26]
  - Often strong for tabular data with careful tuning.

- **XGBRegressor (XGBoost)**
  - Gradient boosting implementation from XGBoost library, typically powerful for structured data.[file:26]
  - Supports regularization and flexible tree growth parameters.

---

## Model Evaluation

The project uses several metrics to evaluate model performance:

- **Mean Squared Error (MSE)**
  - Average of squared prediction errors.[file:26]
- **Root Mean Squared Error (RMSE)**
  - Square root of MSE; in the same units as `audience_count`, making interpretation easier.[file:26]
- **Mean Absolute Error (MAE)**
  - Average absolute error; more robust to outliers than MSE.[file:26]
- **R² Score (Coefficient of Determination)**
  - Measures the proportion of variance in `audience_count` explained by the model.[file:26]

For each model, the notebook:

- Fits the model on the training set.  
- Evaluates on the validation set using these metrics.  
- Collects the results to compare which model performs best in terms of RMSE and related metrics.[file:26]

---

## Hyperparameter Tuning

To avoid relying on default parameters, the notebook uses **RandomizedSearchCV** to tune selected models (such as RandomForestRegressor and XGBRegressor):[file:26]

- Randomly samples combinations of hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`) from specified distributions.[file:26]
- Evaluates each configuration with cross‑validation using RMSE or related metrics.[file:26]
- Selects the best‑performing configuration for final training and evaluation.

This step helps control overfitting while improving predictive performance.

---

## Final Model and Prediction Workflow

The overall workflow to get predictions suitable for Kaggle submission is:

1. Build the enriched training dataset with:
   - Engineered features (date, theater, booking, aggregated audience stats).  
   - Target `audience_count` from `booknow_visits`.[file:26]

2. Train multiple candidate models and compare them based on validation RMSE, MAE, and R².[file:26]

3. Select the **best model configuration** (typically one of the ensemble models such as GradientBoostingRegressor or XGBRegressor) and retrain it on all available labeled data.[file:26]

4. Construct the **test dataset** for future dates:
   - Merge theater information, date features, and booking aggregates for the target prediction period.  
   - Apply the same preprocessing and feature engineering steps as for training.[file:26]

5. Use the final model to generate predicted `audience_count` for each `ID` in `sample_submission.csv`, and export a new `submission.csv` file in the correct format.[file:26]

---

## How This Notebook Achieves the Goal

To summarize the design decisions:

- **Rich feature engineering**:  
  The model does not just rely on raw visits; it combines theater metadata, multi‑source bookings, and calendar features, plus historical per‑theater statistics.[file:26]

- **Model diversity**:  
  Multiple model families (linear, tree‑based, ensembles) are tried, so the final choice is driven by empirical performance, not assumptions.[file:26]

- **Sound evaluation**:  
  Standard metrics (RMSE, MAE, R²) and a clear train/validation split help estimate how well the model will generalize to unseen dates.[file:26]

- **Hyperparameter tuning**:  
  RandomizedSearchCV explores the hyperparameter space efficiently, improving performance over naive defaults and controlling overfitting.[file:26]

- **Reproducible pipeline**:  
  Every step—from raw CSVs to engineered features, model training, and prediction—is encoded in the notebook, making the workflow reproducible for grading or further experimentation.[file:26]

---

## How to Use

1. Open the notebook `22f2001725-notebook-t32025-1.ipynb` in a Kaggle or Jupyter environment with access to the competition data.[file:26]
2. Run all cells from top to bottom:
   - This will load data, perform EDA, construct features, train and evaluate models, and produce final predictions.[file:26]
3. Export the generated `submission.csv` and upload it to the Kaggle **Cinema Audience Forecasting** competition.[file:26]

---

## Future Improvements

Possible future extensions include:

- More advanced time‑series or temporal features (lags, rolling windows).  
- Explicit handling of holidays and special events.  
- Model stacking or blending multiple strong models.  
- Automated feature selection or dimensionality reduction for further robustness.[file:26]
