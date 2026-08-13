# Nassau Candy Factory Optimization Project

## About the Project

This project is about predicting shipping lead time and checking whether changing the factory assigned to a product can improve shipping performance.

The project uses the Nassau Candy dataset and machine learning models to compare different factory scenarios.

## Main Features

- Data cleaning
- Feature engineering
- Exploratory data analysis
- Lead time prediction
- Model comparison
- Route clustering
- Factory scenario simulation
- Factory recommendation
- Streamlit dashboard

## Machine Learning Models

Three models were tested:

- Linear Regression
- Random Forest
- Gradient Boosting

Random Forest performed best among the tested models.

## Model Results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 181.422 | 182.490 | 0.5291 |
| Random Forest | 135.405 | 160.466 | 0.6359 |
| Gradient Boosting | 159.638 | 169.039 | 0.5960 |

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt