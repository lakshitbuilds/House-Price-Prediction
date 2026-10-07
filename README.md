# House Price Prediction

A beginner-friendly Data Science and Machine Learning project that predicts house sale prices using regression models.

## Dataset

Dataset: **House Prices - Advanced Regression Techniques**

Source: Kaggle  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The dataset contains 1,460 training houses and many features describing residential properties in Ames, Iowa. The target column is `SalePrice`.

For this beginner version, only a small set of understandable numerical features is used:

- OverallQual
- GrLivArea
- GarageCars
- TotalBsmtSF
- FullBath
- YearBuilt
- BedroomAbvGr
- TotRmsAbvGrd

## Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation

The models are compared using:

- MAE
- RMSE
- R² Score

## Project Structure

```
House-Price-Prediction/
├── data/
│   └── train.csv
├── notebooks/
│   └── house_price_prediction.ipynb
├── src/
│   └── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

```bash
git clone https://github.com/lakshitbuilds/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
jupyter notebook
```

Open:

```
notebooks/house_price_prediction.ipynb
```

You can also run:

```bash
python src/train_model.py
```

## Workflow

1. Load the dataset
2. Understand columns and data types
3. Check missing values
4. Select useful features
5. Perform basic EDA
6. Split data into training and testing sets
7. Train regression models
8. Evaluate MAE, RMSE, and R²
9. Compare models
10. Predict the price of a new house

## Author

Lakshit Suthar
