from flask import Flask, render_template, request
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


app = Flask(__name__)

import joblib

model = joblib.load("random_forest.pkl")


# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(r"C:\Users\PC\Downloads\demand_forecasting.csv")

print("Dataset loaded successfully!")
print("Rows and Columns:", df.shape)


# =====================================================
# DATE PROCESSING
# =====================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek


# =====================================================
# REMOVE UNNECESSARY COLUMNS
# =====================================================

df_ml = df.drop(
    columns=[
        "Date",
        "Store ID",
        "Product ID",
        "Units Sold",
        "Units Ordered",
        "Inventory Level"
    ],
    errors="ignore"
)


# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df_ml.drop("Demand", axis=1)

y = df_ml["Demand"]


# =====================================================
# CATEGORICAL COLUMNS
# =====================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("Categorical Columns:")
print(categorical_columns)


# =====================================================
# PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# TRAIN MODEL
# =====================================================

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# =====================================================
# MODEL EVALUATION
# =====================================================

y_pred = model.predict(X_test)

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)


print("--------------------------------")
print("MODEL PERFORMANCE")
print("--------------------------------")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 2))
print("--------------------------------")


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        r2_score=round(r2 * 100, 2)
    )


# =====================================================
# PREDICTION
# =====================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Product information

        category = request.form["category"]

        region = request.form["region"]

        price = float(
            request.form["price"]
        )

        discount = float(
            request.form["discount"]
        )

        competitor_pricing = float(
            request.form["competitor_pricing"]
        )


        # Market information

        weather = request.form["weather"]

        promotion = request.form["promotion"]

        seasonality = request.form["seasonality"]

        epidemic = request.form["epidemic"]

        # Convert Yes/No values to numbers
        promotion = 1 if promotion == "Yes" else 0
        epidemic = 1 if epidemic == "Yes" else 0  


        # Date information

        year = int(
            request.form["year"]
        )

        month = int(
            request.form["month"]
        )

        day = int(
            request.form["day"]
        )

        day_of_week = int(
            request.form["day_of_week"]
        )


        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================

        input_data = pd.DataFrame({

            "Category": [category],

            "Region": [region],

            "Price": [price],

            "Discount": [discount],

            "Weather Condition": [weather],

            "Promotion": [promotion],

            "Competitor Pricing": [
                competitor_pricing
            ],

            "Seasonality": [seasonality],

            "Epidemic": [epidemic],

            "Year": [year],

            "Month": [month],

            "Day": [day],

            "DayOfWeek": [day_of_week]

        })


        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            round(prediction, 2)
        )


        # =================================================
        # SHOW RESULT
        # =================================================

        return render_template(
            "index.html",
            prediction=prediction,
            r2_score=round(r2 * 100, 2)
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e),
            r2_score=round(r2 * 100, 2)
        )


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

