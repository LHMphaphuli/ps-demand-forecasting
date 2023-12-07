from distutils.errors import PreprocessError
import logging
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, r_regression
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import make_scorer, mean_absolute_error
from prophet import Prophet
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import LSTM, Dense

from src.utils import fix_missing_values

logger = logging.getLogger(__name__)

PROPHET_FEATURES = [
    "DEMAND_9_NEXT_DAY",
    "DEMAND_10_NEXT_DAY",
    "DEMAND_12_NEXT_DAY",
    "DEMAND_13_NEXT_DAY",
    "DEMAND_15_NEXT_DAY",
    "DEMAND_17_NEXT_DAY",
    "DEMAND_19_NEXT_DAY",
    "DEMAND_32_NEXT_DAY",
    "DEMAND_42_NEXT_DAY",
    "DEMAND_43_CURRENT_DAY",
    "DEMAND_43_NEXT_DAY",
    "DEMAND_44_NEXT_DAY",
    "DEMAND_45_NEXT_DAY",
    "DEMAND_46_NEXT_DAY",
    "PCA4",
    "PCA5",
    "PCA10",
    "POWER_STATION_48_PREVIOUS_DAY",
    "TED_DA_FORECAST",
    "WIND_FORECAST_3_NEXT_DAY",
    "WIND_FORECAST_9_NEXT_DAY",
    "WIND_FORECAST_11_NEXT_DAY",
    "WIND_FORECAST_21_NEXT_DAY",
    "WIND_FORECAST_23_NEXT_DAY",
    "WIND_FORECAST_25_NEXT_DAY",
    "WIND_FORECAST_29_NEXT_DAY",
    "WIND_FORECAST_31_NEXT_DAY",
    "WIND_FORECAST_33_NEXT_DAY",
    "WIND_FORECAST_35_NEXT_DAY",
    "WIND_FORECAST_37_NEXT_DAY",
    "WIND_FORECAST_47_NEXT_DAY"
]

PROPHET_BASE_FEATS = ["WIND", "DEMAND", "TED_DA_FORECAST", "INTERCONNECTOR", "POWER_STATION", "REST"]

def train_gam(target, features, min_train=30):
    logger.info("Training prophet model with lots of features")
        
    # select features on which to base PCA
    X = select_features(features)
    
    preprocessor = get_preprocessor()

    # fit on training data
    proph_data = pd.concat([X, target], axis=1).rename({"PS": "y"}, axis=1)
    proph_data = fix_missing_values(proph_data)
    proph_data.index.name = "ds"
    cols = X.columns.tolist() + [f"PCA{i}" for i in tqdm(range(1, 14))]

    test_predictions = []
    nsplits = abs(round((proph_data.index.min() - proph_data.index.max()).days))
    
    tscv = TimeSeriesSplit(n_splits=nsplits)

    for train_index, test_index in tqdm(tscv.split(proph_data)):

        if len(train_index) < min_train:
            continue
    
        X_train = proph_data.drop(columns="y").iloc[train_index]
        y_train = proph_data[["y"]].iloc[train_index]
        X_test = proph_data.drop(columns="y").iloc[test_index]

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=cols, index=X_train.index)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=cols, index=X_test.index)

        prophmodel = get_prophet_model()
        model = prophmodel.fit(pd.concat([X_train, y_train], axis=1).reset_index())
        
        test_predictions.append(model.predict(X_test.reset_index()))
    
    test_predictions = pd.concat(test_predictions)[["ds", "yhat"]]
    result = test_predictions.rename(columns={"ds":"GAS_DAY", "yhat":"Tuned GAM"}).set_index("GAS_DAY")

    prophmodel = get_prophet_model()
    X_train = pd.DataFrame(preprocessor.fit_transform(proph_data.drop(columns="y")), columns=cols)
    X_train["y"] = proph_data["y"].values
    X_train["ds"] = proph_data.index.values
    model = prophmodel.fit(X_train)

    return model, result

def train_linear_PCA(target, features, min_train=30):
    logger.info("Training linear model with PCA features")
        
    # select features on which to base PCA
    X = select_features(features)
    
    preprocessor = get_preprocessor()
   
    # perform PCA analysis
    pca = PCA(n_components=13)
    pca.fit(X)
    X_pca = pca.transform(X)
    pca_cols = [f"PCA{i}" for i in range(1,14)]
    X = pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

    # fit on training data
    train_data = pd.concat([X, target], axis=1).rename({"PS": "y"}, axis=1)
    train_data = fix_missing_values(train_data)
    
    test_predictions = []
    nsplits = abs(round((train_data.index.min() - train_data.index.max()).days))
    
    tscv = TimeSeriesSplit(n_splits=nsplits)

    for train_index, test_index in tqdm(tscv.split(train_data)):

        if len(train_index) < min_train:
            continue
    
        X_train = train_data.drop(columns="y").iloc[train_index]
        y_train = train_data[["y"]].iloc[train_index]
        X_test = train_data.drop(columns="y").iloc[test_index]

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=pca_cols, index=X_train.index)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=pca_cols, index=X_test.index)

        linearmodel = LinearRegression()
        model_linear = linearmodel.fit(X_train, y_train)
        
        test_predictions.append(model_linear.predict(X_test.reset_index()))
    
    test_predictions = pd.concat(test_predictions)
    linear_result = test_predictions.rename("Tuned Linear")

    return model_linear, linear_result


def get_prophet_model():
    result = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    result.add_country_holidays(country_name="UK")

    for regressor in PROPHET_FEATURES:
        result.add_regressor(regressor)

    return result

def get_preprocessor():
    pipe = make_pipeline(
        StandardScaler(),
        FeatureUnion(
            transformer_list=[
                ('pca', PCA(n_components=13)),
                ('identity', 'passthrough')
            ]
        ),
        StandardScaler()
    )
    return pipe


def select_features(input_data):
    base_features = ["WIND_FORECAST", "INTERCONNECTORS", "REST", "POWER_STATION", "WIND", "DEMAND", "DEMAND"]
    suffix = ["NEXT_DAY", "PREVIOUS_DAY", "PREVIOUS_DAY", "PREVIOUS_DAY","PREVIOUS_DAY", "NEXT_DAY", "CURRENT_DAY"]
    string_mask = [f"{a}_\d*_{b}" for a, b in zip(base_features, suffix)] # i.e. WIND_FORECAST_\d*_NEXT_DAY
    
    mask = input_data.columns.str.contains("|".join(string_mask))
    selected_columns = input_data.columns[mask].tolist()
    result = input_data[selected_columns + ["TED_DA_FORECAST"]].copy()

    # write selected features to file
    output_dir = "data" # modify this to the path of the existing data directory
    output_file = os.path.join(output_dir, "selected_features.csv")
    result.to_csv(output_file, index=False)
    
    return result

def train_glm_63(target, features):
    """Train a Linear Regression model based on CWV

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        pandas Series: A Series with the predictions from the linear model, named GLM_CWV
    """
    logger.info("Training linear model with TED forecast, Wind forecast and Actual within-day so far feature")
    X = features[
        ["TED_DA_FORECAST", "WIND_FORECAST", "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT"]
    ].dropna()

    X, y = check_overlapping_dates(target, X)
    
    model = LinearRegression()
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "GLM_63 NGT Model with 3 Features"
    model.fit(X, y)

    return model, predictions

def train_glm_63_f(target, features):
    """Train a Linear Regression model based on CWV

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best Ridge Regression model and its predictions
    """

    logger.info("Training linear model with TED forecast, Wind forecast, and Actual within-day so far feature")

    # select features on which to base PCA
    X = select_features(features)
    
    preprocessor = get_preprocessor()

    # fit on training data
    glm_data = pd.concat([X, target], axis=1).rename({"PS": "y"}, axis=1)
    glm_data = fix_missing_values(glm_data)
    glm_data.index.name = "ds"
    cols = X.columns.tolist() + [f"PCA{i}" for i in tqdm(range(1, 14))]

    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_1_PREVIOUS_DAY",
            "POWER_STATION_33_PREVIOUS_DAY",
            "POWER_STATION_38_PREVIOUS_DAY",
            "POWER_STATION_43_PREVIOUS_DAY",
            "POWER_STATION_44_PREVIOUS_DAY",  
            "POWER_STATION_16_PREVIOUS_DAY",          
            "POWER_STATION_47_PREVIOUS_DAY",  
            "POWER_STATION_48_PREVIOUS_DAY",                    
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_19_NEXT_DAY",
            "DEMAND_38_NEXT_DAY",
            "DEMAND_43_NEXT_DAY",  
            "DEMAND_45_NEXT_DAY",            
            "DEMAND_46_NEXT_DAY",            
            "DEMAND_48_NEXT_DAY",
            "WIND_FORECAST",
            "WIND_11_PREVIOUS_DAY",
            "REST_7_PREVIOUS_DAY",
            "REST_18_PREVIOUS_DAY",
            "REST_37_PREVIOUS_DAY",
            "REST_1_PREVIOUS_DAY",
            "INTERCONNECTORS_48_PREVIOUS_DAY",
            "INTERCONNECTORS_47_PREVIOUS_DAY",
            "INTERCONNECTORS_11_PREVIOUS_DAY",
            "INTERCONNECTORS_25_PREVIOUS_DAY",
            "INTERCONNECTORS_46_PREVIOUS_DAY",                   
            "DEMAND_11_CURRENT_DAY",
            "DEMAND_19_CURRENT_DAY",
            "DEMAND_47_CURRENT_DAY",
            "DEMAND_38_CURRENT_DAY",
            "DEMAND_48_CURRENT_DAY"            
        ]
    ].dropna()

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    # Create a LinearRegression model
    model = LinearRegression()

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, model)

    # Set the name of the predictions
    predictions.name = "GLM_63 Model with 33 Features"

    # Fit the model to the data
    model.fit(X, y)

    return model, predictions


def train_huber(target, features):
    """Train an improved Huber regression model based on CWV.

    This function trains a Huber regression model on the given data using various improvement techniques such as hyperparameter tuning, feature selection, and regularization. The model is trained using the following hyperparameters:

    * epsilon: The threshold at which the Huber loss function switches from quadratic to linear.
    * alpha: The regularization strength.
    * max_iter: The maximum number of iterations to run the solver.
    * tol: The tolerance for the stopping criteria.

    The predictions of the model are also returned.

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best HuberRegressor model and its predictions
    """

    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_1_PREVIOUS_DAY",
            "POWER_STATION_33_PREVIOUS_DAY",
            "POWER_STATION_38_PREVIOUS_DAY",
            "POWER_STATION_43_PREVIOUS_DAY",
            "POWER_STATION_44_PREVIOUS_DAY",  
            "POWER_STATION_16_PREVIOUS_DAY",          
            "POWER_STATION_47_PREVIOUS_DAY",  
            "POWER_STATION_48_PREVIOUS_DAY",                    
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_19_NEXT_DAY",
            "DEMAND_38_NEXT_DAY",
            "DEMAND_43_NEXT_DAY",  
            "DEMAND_45_NEXT_DAY",            
            "DEMAND_46_NEXT_DAY",            
            "DEMAND_48_NEXT_DAY",
            "WIND_FORECAST",
            "WIND_11_PREVIOUS_DAY",
            "REST_7_PREVIOUS_DAY",
            "REST_18_PREVIOUS_DAY",
            "REST_37_PREVIOUS_DAY",
            "REST_1_PREVIOUS_DAY",
            "INTERCONNECTORS_48_PREVIOUS_DAY",
            "INTERCONNECTORS_47_PREVIOUS_DAY",
            "INTERCONNECTORS_11_PREVIOUS_DAY",
            "INTERCONNECTORS_25_PREVIOUS_DAY",
            "INTERCONNECTORS_46_PREVIOUS_DAY",                   
            "DEMAND_11_CURRENT_DAY",
            "DEMAND_19_CURRENT_DAY",
            "DEMAND_47_CURRENT_DAY",
            "DEMAND_38_CURRENT_DAY",
            "DEMAND_48_CURRENT_DAY"            
        ]
    ].dropna()

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    logger.info("Training an improved Huber regression model with TED forecast, Wind forecast, and Actual within-day so far feature")

    # Increase the number of iterations to 1000 to avoid the convergence warning
    param_grid = {
        'epsilon': [1.25, 1.5, 2, 2.5],
        'alpha': [0.001, 0.005, 0.01],
        'max_iter': [1000, 3000, 5000],
        'tol': [0.01, 0.001, 0.0001]
    }

    # Perform GridSearchCV with TimeSeriesSplit
    cv_method = TimeSeriesSplit(n_splits=5)
    scoring_metric = mean_absolute_error
    predictions = []

    for train_idx, test_idx in cv_method.split(X):
        # Slice the X and y data for train and test sets
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        grid_search = GridSearchCV(
            estimator=HuberRegressor(),
            param_grid=param_grid,
            n_jobs=-1,
            cv=5,  # Reduce the cv parameter as we are now iterating over TimeSeriesSplit object
            scoring=scoring_metric,
        )
        grid_search.fit(X_train, y_train)

        # Get the best model and predictions
        best_params = grid_search.best_params_
        print(f"CV method: TimeSeriesSplit")
        print(f"Best hyperparameters: {best_params}")
          
        best_model = HuberRegressor(**best_params)
        best_model.fit(X_train, y_train)
        predictions.append(best_model.predict(X_test))

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, best_model)
    
    best_model.fit(X_train, y_train)
    
    # Set the name of the predictions
    predictions.name = "Huber Model with 33 Features"
    
    return best_model, predictions


def train_ridge(target, features):
    """Train an improved Ridge regression model based on CWV.

    This function trains a Ridge regression model on the given data using various improvement techniques such as hyperparameter tuning, feature selection, and regularization. The model is trained using the following hyperparameters:

    * alpha: The regularization strength.
    * max_iter: The maximum number of iterations to run the solver.
    * tol: The tolerance for the stopping criteria.

    The predictions of the model are also returned.

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best Ridge model and its predictions
    """

    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_48_PREVIOUS_DAY",
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",  
            "DEMAND_17_NEXT_DAY",            
            "DEMAND_19_NEXT_DAY",            
            "DEMAND_44_NEXT_DAY",
            "WIND_FORECAST_23_NEXT_DAY",
            "WIND_FORECAST_9_NEXT_DAY",
            "WIND_FORECAST_21_NEXT_DAY",
            "WIND_FORECAST_25_NEXT_DAY",
            "WIND_FORECAST_31_NEXT_DAY",
            "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    logger.info("Training an improved Ridge regression model with 14 Features")

    # Increase the number of iterations to 1000 to avoid the convergence warning
    param_grid = {
        'alpha': [0.001, 0.005, 0.01],
        'max_iter': [1000, 2000, 3000],
        'tol': [0.01, 0.001, 0.0001]
    }

    # Perform GridSearchCV with TimeSeriesSplit
    cv_method = TimeSeriesSplit(n_splits=5)
    scoring_metric = mean_absolute_error
    predictions = []

    for train_idx, test_idx in cv_method.split(X):
        # Slice the X and y data for train and test sets
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        grid_search = GridSearchCV(
            estimator=Ridge(),
            param_grid=param_grid,
            n_jobs=-1,
            cv=3,  # Reduce the cv parameter as we are now iterating over TimeSeriesSplit object
            scoring=scoring_metric,
        )
        grid_search.fit(X_train, y_train)

        # Get the best model and predictions
        best_params = grid_search.best_params_
        print(f"CV method: TimeSeriesSplit")
        print(f"Best hyperparameters: {best_params}")
          
        best_model = Ridge(**best_params)
        best_model.fit(X_train, y_train)
        predictions.append(best_model.predict(X_test))

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, best_model)
    
    best_model.fit(X_train, y_train)
    
    # Set the name of the predictions
    predictions.name = "Ridge Model with 14 Features"
    
    return best_model, predictions


def train_theil_sen(target, features):
    """Train a Theil-Sen regressor model based on CWV.

    This function trains a Theil-Sen regressor model based on the given data using various improvement techniques such as hyperparameter tuning and feature selection.

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best TheilSenRegressor model and its predictions
    """
    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_48_PREVIOUS_DAY",
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",  
            "DEMAND_17_NEXT_DAY",            
            "DEMAND_19_NEXT_DAY",            
            "DEMAND_43_NEXT_DAY",               
            "DEMAND_44_NEXT_DAY",   
            "WIND_FORECAST_9_NEXT_DAY",           
            "WIND_FORECAST_21_NEXT_DAY",
            "WIND_FORECAST_23_NEXT_DAY", 
            "WIND_FORECAST_25_NEXT_DAY",
            "WIND_FORECAST_31_NEXT_DAY",            
            "WIND_FORECAST_37_NEXT_DAY",
        ]
    ].dropna()

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    logger.info("Training a Theil-Sen Regressor model with TED forecast, Wind forecast, and Actual within-day so far feature")

    # Set the parameter grid for the Theil-Sen regressor
    param_grid = {
        'fit_intercept': [True, False],
        'tol': [1e-3, 1e-4, 1e-5, 1e-6],
        'max_subpopulation': [10000],
        'max_iter': [10000, 15000]
    }

    # Perform RandomizedSearchCV with different cross-validation methods and scoring metrics
    cv_method = KFold(n_splits=10, shuffle=True, random_state=42)
    scoring_metric = mean_absolute_error
    predictions = []

    for train_idx, test_idx in cv_method.split(X):
        # Slice the X and y data for train and test sets
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        random_search = RandomizedSearchCV(
            estimator=TheilSenRegressor(),
            param_distributions=param_grid,
            n_jobs=-1,
            cv=3,
            verbose=4,
            n_iter=10, # set this based on your computation power and time budget
            scoring=scoring_metric
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        predictions.append(best_model.predict(X_test))

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, best_model)

    best_model.fit(X, y)

    # Set the name of the predictions
    predictions.name = "Theil-Sen Model with 15 Features"

    return best_model, predictions

def train_stacking_reg(target, features):
    """
    Train a linear-huber stacking regressor model with 14 features based on CWV and return the model and its predictions.
    """

    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_48_PREVIOUS_DAY",
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",  
            "DEMAND_17_NEXT_DAY",            
            "DEMAND_19_NEXT_DAY",            
            "DEMAND_44_NEXT_DAY",
            "WIND_FORECAST_23_NEXT_DAY",
            "WIND_FORECAST_9_NEXT_DAY",
            "WIND_FORECAST_21_NEXT_DAY",
            "WIND_FORECAST_25_NEXT_DAY",
            "WIND_FORECAST_31_NEXT_DAY",
            "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()

    if X.empty:
        raise ValueError("Empty feature matrix after dropping NaN values")

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    logger.info("Performing hyperparameter tuning for the Huber regressor")
    # Set up grid search parameters and perform RandomizedSearchCV
    param_grid = {
        'epsilon': [1.0, 1.25, 1.5],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [1000, 2000, 3000],
        'tol': [0.001, 0.0001]
    }

    cv_method = TimeSeriesSplit(n_splits=5)
    scoring_metric = mean_absolute_error
    grid_search = RandomizedSearchCV(
        estimator=HuberRegressor(),
        param_distributions=param_grid,
        n_jobs=-1,
        cv=cv_method,
        scoring=scoring_metric,
        n_iter=30,  
        random_state=42
    )
    grid_search.fit(X, y)

    # Get the best hyperparameters and log the results
    best_params = grid_search.best_params_
    logger.info(f"Best hyperparameters: {best_params}")
    
  
    # Train the linear-huber stacking regressor model
    estimators = [
        ('lr', LinearRegression()),
        ('huber', HuberRegressor(**best_params))
    ]

    huber_regressor = HuberRegressor(**best_params)

    stack_reg = StackingRegressor(
        estimators=estimators, final_estimator=huber_regressor
    )

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, stack_reg)
    stack_reg.fit(X, y)
    predictions.name = "Linear-Huber Stacking Regressor model with 14 features"

    return stack_reg, predictions



def train_ransac(target, features):
    """Train a RANSAC regression model based on GLM_63 linear regression.

    This function trains a RANSAC regression model on the given data using the GLM_63 linear regression model as the base estimator. The features used in this function are:
    ['TED_DA_FORECAST', 'ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT', 'POWER_STATION_48_PREVIOUS_DAY', 'DEMAND_9_NEXT_DAY', 'DEMAND_10_NEXT_DAY', 'DEMAND_17_NEXT_DAY', 'DEMAND_19_NEXT_DAY', 'DEMAND_44_NEXT_DAY', 'WIND_FORECAST_23_NEXT_DAY', 'WIND_FORECAST_9_NEXT_DAY', 'WIND_FORECAST_21_NEXT_DAY', 'WIND_FORECAST_25_NEXT_DAY', 'WIND_FORECAST_31_NEXT_DAY', 'WIND_FORECAST_37_NEXT_DAY']

    The model is trained using the following hyperparameters:

    * min_samples: The minimum number of samples required to consider a data point as an inlier.
    * max_trials: The maximum number of iterations to run the RANSAC algorithm.
    * residual_threshold: The maximum distance between the predicted value and the true value.
    * linear_model: The base estimator to use for fitting the inliers during each iteration of the RANSAC algorithm.

    The predictions of the model are also returned.

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best RANSACRegressor model and its predictions
    """
    
    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_1_PREVIOUS_DAY",
            "POWER_STATION_33_PREVIOUS_DAY",
            "POWER_STATION_38_PREVIOUS_DAY",
            "POWER_STATION_43_PREVIOUS_DAY",
            "POWER_STATION_44_PREVIOUS_DAY",  
            "POWER_STATION_16_PREVIOUS_DAY",          
            "POWER_STATION_47_PREVIOUS_DAY",  
            "POWER_STATION_48_PREVIOUS_DAY",                    
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_19_NEXT_DAY",
            "DEMAND_38_NEXT_DAY",
            "DEMAND_43_NEXT_DAY",  
            "DEMAND_45_NEXT_DAY",            
            "DEMAND_46_NEXT_DAY",            
            "DEMAND_48_NEXT_DAY",
            "WIND_FORECAST",
            "WIND_11_PREVIOUS_DAY",
            "REST_7_PREVIOUS_DAY",
            "REST_18_PREVIOUS_DAY",
            "REST_37_PREVIOUS_DAY",
            "REST_1_PREVIOUS_DAY",
            "INTERCONNECTORS_48_PREVIOUS_DAY",
            "INTERCONNECTORS_47_PREVIOUS_DAY",
            "INTERCONNECTORS_11_PREVIOUS_DAY",
            "INTERCONNECTORS_25_PREVIOUS_DAY",
            "INTERCONNECTORS_46_PREVIOUS_DAY",                   
            "DEMAND_11_CURRENT_DAY",
            "DEMAND_19_CURRENT_DAY",
            "DEMAND_47_CURRENT_DAY",
            "DEMAND_38_CURRENT_DAY",
            "DEMAND_48_CURRENT_DAY"            
        ]
    ].dropna()

    logger.info("Training a RANSAC regression model based on GLM_63 linear regression with 14 features")

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    # Define the parameter grid to search over
    param_grid = {
        'min_samples': [50, 100, 150],
        'max_trials': [100, 300, 600],
        'residual_threshold': [2, 3, 4],
    }

    # Define the mean absolute error scorer
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Define the GridSearchCV object
    grid_search = GridSearchCV(RANSACRegressor(), param_grid=param_grid, scoring={'r2': 'r2', 'mae': mae_scorer}, refit='r2', cv=5)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X, y)

    # Print the best hyperparameters and evaluation metrics
    logger.info("Best hyperparameters: {}".format(grid_search.best_params_))
    logger.info("Best R2 score: {}".format(grid_search.best_score_))
    logger.info("Best mean absolute error: {}".format(-grid_search.cv_results_['mean_test_mae'][grid_search.best_index_]))
    
    # Train the RANSAC regressor using the best hyperparameters
    ransac_best = grid_search.best_estimator_
    ransac_best.fit(X, y)

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, ransac_best)

    # Set the name of the predictions
    predictions.name = "RANSAC Model with 33 Features"

    return ransac_best, predictions



def train_lgbm(target, features):
    """Train a LightGBM model based on Regression approach

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best LightGBM model and its predictions
    """
    logger.info("Training LightGBM model with TED forecast, Wind forecast, and Actual within-day so far feature")

    # This function trains a LightGBM regression model on the given data.
    # The model is trained using hyperparameter tuning to select the best parameters.
    # The predictions of the model are also returned.


    # Define the feature matrix
    # The feature matrix is a DataFrame that contains the features to be used for training the model.
    # The features are selected based on domain knowledge and previous experiments.
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_48_PREVIOUS_DAY",
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_44_NEXT_DAY",
            "WIND_FORECAST_23_NEXT_DAY",
            "WIND_FORECAST_9_NEXT_DAY",
            "WIND_FORECAST_21_NEXT_DAY",
            "WIND_FORECAST_25_NEXT_DAY",
            "WIND_FORECAST_31_NEXT_DAY",
            "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()

    # Check overlapping dates
    # This function checks if the dates in the target and features DataFrames overlap.
    # If the dates do not overlap, the DataFrames are aligned so that they do.

    X, y = check_overlapping_dates(target, X)
    
    # Define hyperparameters to be tuned
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Use grid search to find the best hyperparameters for the model
    grid_search = GridSearchCV(
        lgb.LGBMRegressor(),
        param_grid=param_grid,
        n_jobs=-1,
        cv=10,
        verbose=4,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False)
    )

    grid_search.fit(X, y)

    # Print the best parameters found by grid search
    print(grid_search.best_params_)

    # Train the model with the best hyperparameters
    model = lgb.LGBMRegressor(**grid_search.best_params_)
    model.fit(X, y)

    # Make predictions on the test set
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "LightGBM Regression"

    return model, predictions


def train_xgboost(target, features):
    """Train an XGBoost model based on Regression approach

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best XGBoost model and its predictions
    """

    # This function trains an XGBoost regression model on the given data.
    # The model is trained using hyperparameter tuning to select the best parameters.
    # The predictions of the model are also returned.


    # Logging the start of training
    logger.info("Training XGBoost model with TED forecast, Wind forecast, and Actual within-day so far feature")

    # Define the feature matrix
    # The feature matrix is a DataFrame that contains the features to be used for training the model.
    # The features are selected based on domain knowledge and previous experiments.

    X = features[
        [
        "TED_DA_FORECAST", 
        "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT", 
        "POWER_STATION_48_PREVIOUS_DAY", 
        "DEMAND_9_NEXT_DAY",
        "DEMAND_10_NEXT_DAY",        
        "DEMAND_44_NEXT_DAY",
        "WIND_FORECAST_23_NEXT_DAY",
        "WIND_FORECAST_9_NEXT_DAY",        
        "WIND_FORECAST_21_NEXT_DAY",
        "WIND_FORECAST_25_NEXT_DAY",
        "WIND_FORECAST_31_NEXT_DAY",
        "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()

    # Check overlapping dates
    # This function checks if the dates in the target and features DataFrames overlap.
    # If the dates do not overlap, the DataFrames are aligned so that they do.

    X, y = check_overlapping_dates(target, X)

    # Create a parameter grid for the XGBoost regressor
    param_grid = {
        "n_estimators": [100, 250, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5],
        "min_child_weight": [5],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
        "reg_lambda": [0, 1, 10, 100],
        "reg_alpha": [0, 1, 10, 100],
    }

    # Use grid search to find the best hyperparameters for the model
    grid_search = GridSearchCV(
        xgb.XGBRegressor(),
        param_grid=param_grid,
        n_jobs=-1,
        cv=10,
        verbose=4,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False)
    )

    grid_search.fit(X, y)

    # Print the best parameters found by grid search
    print(grid_search.best_params_)

    # Train the model with the best hyperparameters
    model = xgb.XGBRegressor(**grid_search.best_params_)
    model.fit(X, y)

    # Make predictions on the test set
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "XGBoost Regression"

    return model, predictions


def train_gbt(target, features):
    """Train a Gradient Boosted Trees Regressor model based on the features and target variables

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best Gradient Boosted Trees Regressor model and its predictions
    """
    logger.info("Training Gradient Boosted Trees Regressor model with TED forecast, Demand Forecast and Wind forecast")
    
    # This function trains a gradient boosted trees regression model on the given data.
    # The model is trained using hyperparameter tuning to select the best parameters.
    # The predictions of the model are also returned.


    # Define the feature matrix
    # The feature matrix is a DataFrame that contains the features to be used for training the model.
    # The features are selected based on domain knowledge and previous experiments.

    X = features[
        [
        "TED_DA_FORECAST", 
        "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT", 
        "POWER_STATION_48_PREVIOUS_DAY", 
        "DEMAND_9_NEXT_DAY",
        "DEMAND_10_NEXT_DAY",        
        "DEMAND_44_NEXT_DAY",
        "WIND_FORECAST_23_NEXT_DAY",
        "WIND_FORECAST_9_NEXT_DAY",        
        "WIND_FORECAST_21_NEXT_DAY",
        "WIND_FORECAST_25_NEXT_DAY",
        "WIND_FORECAST_31_NEXT_DAY",
        "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()
  
    # Check overlapping dates
    # This function checks if the dates in the target and features DataFrames overlap.
    # If the dates do not overlap, the DataFrames are aligned so that they do.

    X, y = check_overlapping_dates(target, X)

    # Create a parameter grid for the gradient boosted trees regressor
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [50, 100, 200],
        'max_depth': [4],
        'min_samples_split': [5, 10],
    }

    # Use grid search to find the best hyperparameters for the model
    grid_search = GridSearchCV(
        GradientBoostingRegressor(),
        param_grid=param_grid,
        n_jobs=-1,
        cv=10,
        verbose=4,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False)
    )

    grid_search.fit(X, y)

    # Print the best parameters found by grid search
    print(grid_search.best_params_)

    # Train the model with the best hyperparameters
    model = GradientBoostingRegressor(**grid_search.best_params_)
    model.fit(X, y)

    # Make predictions on the test set using time series cross-validation
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "Gradient Boosted Trees Regressor"

    return model, predictions


def train_random_forest(target, features):
    """Train a Random Forest Regressor model with tuning of its hyperparameters based on the features and target variables

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best Random Forest Regressor model and its predictions
    """
    logger.info("Training Random Forest Regressor model with tuning of its hyperparameters based on TED forecast, Demand Forecast and Wind forecast")
    
    # This function trains a random forest regression model on the given data.
    # The model is trained using hyperparameter tuning to select the best parameters.
    # The predictions of the model are also returned.


    # Define the feature matrix
    # The feature matrix is a DataFrame that contains the features to be used for training the model.
    # The features are selected based on domain knowledge and previous experiments.
    X = features[
        [
        "TED_DA_FORECAST", 
        "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT", 
        "POWER_STATION_48_PREVIOUS_DAY", 
        "DEMAND_9_NEXT_DAY",
        "DEMAND_10_NEXT_DAY",        
        "DEMAND_44_NEXT_DAY",
        "WIND_FORECAST_23_NEXT_DAY",
        "WIND_FORECAST_9_NEXT_DAY",        
        "WIND_FORECAST_21_NEXT_DAY",
        "WIND_FORECAST_25_NEXT_DAY",
        "WIND_FORECAST_31_NEXT_DAY",
        "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()
  
    # Check overlapping dates
    # This function checks if the dates in the target and features DataFrames overlap.
    # If the dates do not overlap, the DataFrames are aligned so that they do.
    X, y = check_overlapping_dates(target, X)

    # Create a parameter grid for the random forest regressor
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [2, 5, 10, 20],
        'min_samples_split': [5, 10],
    }

    # Use grid search to find the best hyperparameters for the model
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        n_jobs=-1,
        cv=10,
        verbose=4,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False)
    )

    grid_search.fit(X, y)

    # Print the best parameters found by grid search
    print(grid_search.best_params_)

    # Train the model with the best hyperparameters
    model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    model.fit(X, y)

    # Make predictions on the test set using time series cross-validation
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "Random Forest Regressor"

    return model, predictions


def train_linear_svr(target, features):
    """Train an SVR model 
    
    Args:
        target (pandas DataFrame): A DataFrame with a column named PS for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast
    
    Returns:
        tuple: A tuple containing the best SVR model and its predictions
    """
    logger.info("Training SVR model with hyperparameter tuning")

    # Define the feature matrix
    # The feature matrix is a DataFrame that contains the features to be used for training the model.
    # The features are selected based on domain knowledge and previous experiments.
    X = features[
        [
        "TED_DA_FORECAST", 
        "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT", 
        "POWER_STATION_48_PREVIOUS_DAY", 
        "DEMAND_9_NEXT_DAY",
        "DEMAND_10_NEXT_DAY",
        "DEMAND_43_NEXT_DAY",               
        "DEMAND_44_NEXT_DAY",
        "WIND_FORECAST_9_NEXT_DAY",           
        "WIND_FORECAST_21_NEXT_DAY",
        "WIND_FORECAST_23_NEXT_DAY", 
        "WIND_FORECAST_25_NEXT_DAY",
        "WIND_FORECAST_31_NEXT_DAY",
        "WIND_FORECAST_37_NEXT_DAY"
        ]
    ].dropna()

    # Check overlapping dates
    # This function checks if the dates in the target and features DataFrames overlap.
    # If the dates do not overlap, the DataFrames are aligned so that they do.
    X, y = check_overlapping_dates(target, X)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0.1, 0.2, 0.3, 0.4],
        'max_iter': [100, 1000, 10000],
    }

    # Get the best model and predictions
    # This function uses hyperparameter tuning to find the best parameters for the SVR model.
    # The best model is then used to make predictions on the test data.
    model = LinearSVR()
    svr_cv = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
    svr_cv.fit(X, y)
    best_model = svr_cv.best_estimator_
    predictions = tss_cross_val_predict(X, y, best_model)
    predictions.name = "Linear SVR Model"

    return best_model, predictions


def tss_cross_val_predict(X, y, model, min_train=30):
    """Apply a form of Time Series cross validation with the given data and for the given model
    We expand the data by a day in each fold and retrain the model to generate predictions for the next day.

    Args:
        X (pandas DataFrame): A DataFrame with features
        y (pandas DataFrame): A DataFrame with the target
        model (a sklearn Model): A model object with a fit and predict function
        min_train (int, optional): Number of historical values necessary to start the training cadence. Defaults to 7.

    Returns:
        pandas Series: A Series with the predictions from all the folds
    """
    test_predictions = []

    # daily window

    nsplits = int(abs((round((X.index.min() - X.index.max()).days)/2)))
    tscv = TimeSeriesSplit(n_splits=nsplits, gap=2)

    for train_index, test_index in tscv.split(X):
        if len(train_index) < min_train:
            continue

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]

        model.fit(X_train, y_train)
        test_predictions.append(model.predict(X_test).flatten())

    test_predictions = np.concatenate(test_predictions)
    test_predictions = pd.Series(
        test_predictions, index=X.index[-len(test_predictions):]
    )

    return test_predictions


def check_overlapping_dates(dataset_one, dataset_two):
    """Determine the overlapping dates from the given datasets and filter them both by it

    Args:
        dataset_one (pandas DataFrame): A DataFrame with dates on the index
        dataset_two (pandas DataFrame): A DataFrame with dates on the index

    Returns:
        tuple: A tuple of DataFrame, in reverse order from the input (just to confuse you)
    """
    overlapping_dates = dataset_one.index.intersection(dataset_two.index)
    d1 = dataset_one.loc[overlapping_dates].copy()
    d2 = dataset_two.loc[overlapping_dates].copy()

    return d2, d1

