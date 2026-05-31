import os
import pickle
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =====================================================
# Create Models Folder
# =====================================================

os.makedirs("models", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

data = load_diabetes()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

print("Dataset Shape :", df.shape)

# =====================================================
# Data Cleaning
# =====================================================

print("\nMissing Values")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates :", df.shape)

# =====================================================
# Outlier Treatment Using IQR
# =====================================================

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# =====================================================
# Feature Engineering
# =====================================================

df["bmi_age"] = df["bmi"] * df["age"]

df["bp_s5"] = df["bp"] * df["s5"]

df["s1_s2_ratio"] = df["s1"] / (df["s2"] + 1e-5)

# =====================================================
# Remove Unnecessary Columns
# =====================================================

df.drop(
    columns=[
        "s1",
        "s2",
        "s3"
    ],
    inplace=True
)

# =====================================================
# Features and Target
# =====================================================

X = df.drop("target", axis=1)

y = df["target"]

print("\nFinal Features")
print(X.columns.tolist())

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Save Scaler

with open("models/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# =====================================================
# Base Learners
# =====================================================

base_models = [

    (
        "lr",
        LinearRegression()
    ),

    (
        "dt",
        DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        )
    ),

    (
        "rf",
        RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
    )
]

# =====================================================
# Meta Learner
# =====================================================

meta_model = LinearRegression()

# =====================================================
# Stacking Regressor
# =====================================================

model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# =====================================================
# Model Training
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# Prediction
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")

print("MAE :", round(mae, 4))

print("MSE :", round(mse, 4))

print("RMSE :", round(rmse, 4))

print("R2 Score :", round(r2, 4))

# =====================================================
# Save Model
# =====================================================

with open("models/stacking_regressor.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel Saved Successfully")