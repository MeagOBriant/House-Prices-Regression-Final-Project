# House-Prices-Regression-Final-Project
Final Project for DATA522 -Predict sales prices and practice feature engineering, RFs, and gradient boosting: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
This project focuses on developing a predictive model for estimating housing sale prices using the Ames Housing dataset provided by Kaggle. The dataset contains 79 explanatory variables describing various attributes of residential homes, including lot size, year built, quality ratings, neighborhood, and more. Predicting housing prices is a highly relevant problem in real estate analytics and serves as an excellent opportunity to explore advanced regression models and feature engineering techniques. The objective is to apply machine learning methods to accurately predict house prices and evaluate model performance using Root Mean Square Error (RMSE), which serves as the competition’s scoring metric.

Strategy: 

Data Exploration & Cleaning:
    Understand data structure and handle missing values using imputation or removal.
    Visualize feature distributions and check for outliers.

Feature Engineering:
    Transform skewed numerical features (e.g., log transformation).
    Create new interaction terms or categorical binning if necessary.
    One-hot encode categorical variables.

Modeling Techniques:
    Begin with baseline Linear Regression to establish a reference.
    Apply more advanced models such as Random Forest, XGBoost, and LightGBM.
    Use cross-validation to tune hyperparameters and avoid overfitting.

Model Evaluation:
    Compare models using cross-validated RMSE scores.
    Select the best-performing model based on validation performance.

Tools:
    Python (Jupyter Notebook), Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib/Seaborn for EDA.
Version Control & Collaboration:

    Project files will be maintained on GitHub (to be linked in final submission).
