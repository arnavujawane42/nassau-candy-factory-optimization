import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/nassau_candy_features.csv"

df = pd.read_csv(file_path)

print("Feature dataset loaded successfully!")

print("Rows:", len(df))


# ==========================================
# 2. SELECT FEATURES
# ==========================================

categorical_features = [
    "Product Name",
    "Current Factory",
    "Region",
    "Ship Mode",
    "Division"
]


numerical_features = [
    "Units",
    "Sales",
    "Cost",
    "Gross Profit",
    "Order Year",
    "Order Month",
    "Order Day",
    "Order Day of Week"
]


features = (
    categorical_features +
    numerical_features
)


target = "Lead Time"


X = df[features]

y = df[target]


print("\n====================================")
print("FEATURES USED")
print("====================================")

print(features)

print("\nTarget:", target)


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n====================================")
print("DATA SPLIT")
print("====================================")

print("Training records:", len(X_train))

print("Testing records:", len(X_test))


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            StandardScaler(),

            numerical_features
        )
    ]
)


# ==========================================
# 5. CREATE MODELS
# ==========================================

models = {

    "Linear Regression":

        LinearRegression(),


    "Random Forest":

        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ==========================================
# 6. TRAIN MODELS
# ==========================================

results = []


for model_name, model in models.items():

    print("\n====================================")

    print(model_name)

    print("====================================")


    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    # Train model

    pipeline.fit(
        X_train,
        y_train
    )


    # Make predictions

    predictions = pipeline.predict(
        X_test
    )


    # MAE

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    # RMSE

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    # R2

    r2 = r2_score(
        y_test,
        predictions
    )


    print(
        "MAE :",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )

    print(
        "R2  :",
        round(r2, 4)
    )


    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)


print("\n====================================")

print("MODEL COMPARISON")

print("====================================")


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

best_model_name = (

    results_df

    .sort_values("RMSE")

    .iloc[0]["Model"]
)


print("\n====================================")

print("BEST MODEL")

print("====================================")


print(
    "Best model based on RMSE:",
    best_model_name
)


# ==========================================
# 9. TRAIN BEST MODEL
# ==========================================

best_model = models[
    best_model_name
]


best_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            best_model
        )
    ]
)


best_pipeline.fit(
    X_train,
    y_train
)


# ==========================================
# 10. SAVE BEST MODEL
# ==========================================

joblib.dump(

    best_pipeline,

    "best_lead_time_model.pkl"
)


print("\n====================================")

print("MODEL SAVED")

print("====================================")


print(
    "Saved as: best_lead_time_model.pkl"
)


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results_df.to_csv(

    "data/model_results.csv",

    index=False
)


print(
    "Model results saved to:",
    "data/model_results.csv"
)


print("\n====================================")

print("MACHINE LEARNING COMPLETE")

print("====================================")