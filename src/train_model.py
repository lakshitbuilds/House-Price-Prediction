from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"
df = pd.read_csv(data_path)

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
]

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

best_model = None
best_rmse = float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(name)
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 3))
    print()

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

new_house = pd.DataFrame([{
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "TotalBsmtSF": 1000,
    "FullBath": 2,
    "YearBuilt": 2005,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 7,
}])

predicted_price = best_model.predict(new_house)[0]
print("Predicted House Price:", round(predicted_price, 2))
