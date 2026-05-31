import os
import pickle
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from sklearn.metrics import r2_score

os.makedirs("models", exist_ok=True)

# ==================================================
# Load Dataset
# ==================================================

data = load_diabetes()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

# ==================================================
# Data Cleaning
# ==================================================

df.drop_duplicates(inplace=True)

# ==================================================
# Feature Engineering
# ==================================================

df["bmi_age"] = df["bmi"] * df["age"]

df["bp_s5"] = df["bp"] * df["s5"]

df["s1_s2_ratio"] = df["s1"] / (df["s2"] + 1e-5)

# ==================================================
# Features and Target
# ==================================================

X = df.drop("target", axis=1)
y = df["target"]

# ==================================================
# Train Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# Scaling
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

# ==================================================
# Base Models
# ==================================================

base_models = [

    ("lr", LinearRegression()),

    ("dt", DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    )),

    ("rf", RandomForestRegressor(
        n_estimators=10,
        max_depth=5,
        random_state=42
    ))
]

# ==================================================
# Meta Learner
# ==================================================

meta_model = LinearRegression()

# ==================================================
# Stacking Regressor
# ==================================================

model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

model.fit(X_train, y_train)

# ==================================================
# Evaluation
# ==================================================

y_pred = model.predict(X_test)

print("R2 Score :", round(r2_score(y_test, y_pred), 4))

# ==================================================
# Save Model
# ==================================================

pickle.dump(
    model,
    open("models/stacking_regressor.pkl", "wb")
)

print("Model Saved Successfully")