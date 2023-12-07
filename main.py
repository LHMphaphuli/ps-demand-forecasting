import joblib
import logging
import os
from datetime import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_electricity_features,
)
from src.train import train_glm_63_f, train_glm_63, train_ransac, train_huber

#from src.train import train_glm_63, train_gam, train_glm_63_f, train_huber, train_lgbm,  train_xgboost, train_linear_svr, train_stacking_regressor, train_gbt, train_random_forest, train_ransac, train_theil_sen

from src.evaluate import evaluate_models

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

FORMAT = "%Y%m%d_%H%M%S"

FEATURES = {
    "TED": "data/elexon_ted_forecast_20230909_220251.csv",
    "WIND": "data/elexon_wind_forecast_20230909_220400.csv",
    "ACTUAL_D_SOFAR_ALL_BUT_WIND_GT": "data/elexon_electricity_actuals_20230909_220522.csv",
    "ELECTRICITY_ACTUALS": "data/elexon_electricity_actuals_20230909_220522.csv",
}
ACTUALS = {"GAS": "data/gas_actuals_20230909_220003.csv"}

logger.info("Preprocessing actual gas demand")
gas_demand_actuals = prepare_gas_demand_actuals(ACTUALS["GAS"])
ps_demand_actuals = gas_demand_actuals[["PS"]]

logger.info("Preparing features")
electricity_features = prepare_electricity_features(FEATURES)

# This code trains a generalized linear model (GLM) with 3 features.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_63_model, ps_63_cv_predictions = train_glm_63(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_63_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

# This code trains a GLM with original features and added features.
# The added features are the day of the week, the hour of the day, and the month of the year.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_63_f_model, ps_63_f_cv_predictions = train_glm_63_f(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_63_f_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")
"""
# This code trains a Theil-Sen model.
# The Theil-Sen model is a robust regression model that is not sensitive to outliers.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_theil_sen_model, ps_theil_sen_cv_predictions = train_theil_sen(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_theil_sen_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")
"""
# This code trains a RANSAC model.
# The RANSAC model is a robust regression model that is also not sensitive to outliers.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_ransac_model, ps_ransac_cv_predictions = train_ransac(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_ransac_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

# This code trains a Huber model.
# The Huber model is a robust regression model that is more sensitive to outliers than the Theil-Sen or RANSAC models.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_huber_model, ps_huber_cv_predictions = train_huber(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_huber_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")
"""
# This code trains a Ridge model.
# The Ridge model is a robust regression model that is more sensitive to outliers
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_ridge_model, ps_ridge_cv_predictions = train_ridge(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_ridge_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")


# This code trains a stacking regressor model.
# A stacking regressor is a meta-model that combines the predictions of multiple other models.
# The models that are combined in the stacking regressor are the GLM with 63 features, and the Huber model.
# The stacking regressor is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_stacking_reg_model, ps_stacking_reg_cv_predictions = train_stacking_reg(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_stacking_reg_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

# This code trains a XGBoost model to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_xgboost_model, ps_xgboost_cv_predictions = train_xgboost(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_xgboost_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")
"""
"""
# This code trains a gradient boosted trees (GBT) model to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_gbt_model, ps_gbt_cv_predictions = train_gbt(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_gbt_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

# This code trains a random forest model to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_random_forest_model, ps_random_forest_cv_predictions = train_random_forest(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_random_forest_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

# This code trains a light gradient boosting machine (LGBM) model to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_lgbm_model, ps_lgbm_cv_predictions = train_lgbm(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_lgbm_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")
"""
# This code creates an ensemble model by averaging the predictions of the GLM_63_Added_Ftrs and Huber models.
ps_ensemble_63f_huber_cv_predictions = (0.5*ps_63_f_cv_predictions) + (0.5*ps_huber_cv_predictions)
ps_ensemble_63f_huber_cv_predictions.name = "GLM_63 & Huber Models Ensemble with 33 Features"
"""

# This code trains a generalized additive model (GAM) to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_linear_PCA_model, ps_linear_PCA_cv_predictions = train_linear_PCA(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_linear_PCA_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

# This code trains a SARIMA model to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_lstm_model, ps_lstm_cv_predictions = train_lstm(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_lstm_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")
"""   
# Concatenate all predictions
all_predictions = pd.concat(
    [
    ps_63_cv_predictions.to_frame(),
    ps_63_f_cv_predictions.to_frame(),
    #ps_xgboost_cv_predictions.to_frame(),
    #ps_gbt_cv_predictions.to_frame(),
    #ps_random_forest_cv_predictions.to_frame(),
    #ps_lstm_cv_predictions.to_frame()
    #ps_theil_sen_cv_predictions.to_frame(),
    #ps_ridge_cv_predictions.to_frame(),
    ps_huber_cv_predictions.to_frame(),
    ps_ransac_cv_predictions.to_frame(),
    ps_ensemble_63f_huber_cv_predictions.to_frame(),
    #ps_stacking_reg_cv_predictions.to_frame(),
    #ps_lgbm_cv_predictions.to_frame(),
    #ps_linear_PCA_cv_predictions
    ],
axis=1,
)


# This code evaluates the performance of the ensemble model using mean absolute error (MAE) and mean absolute percentage error (MAPE).
# The MAE is the average difference between the predicted and actual values.
# The MAPE is the average percentage error between the predicted and actual values.
model_performance = evaluate_models(all_predictions, ps_demand_actuals)

# This code calculates the MAE deviation percentage for the ensemble model.
# The MAE deviation percentage is the difference between the MAE of the ensemble model and the MAE of the GLM_63 model, expressed as a percentage of the MAE of the GLM_63 model.
glm63_mae = model_performance.loc[model_performance['MODEL'] == 'GLM_63 NGT Model with 3 Features', 'MAE'].iloc[0]
model_performance['MAE_Deviation %'] = ((model_performance['MAE'] - glm63_mae) / glm63_mae) * 100

# This code sorts the models by MAE.
# The models are then ranked by their MAE and MAPE scores.
model_performance = model_performance.sort_values(by='MAE')
model_performance['MAE Ranking'] = model_performance['MAE'].rank(method='min').astype(int)
model_performance['MAPE Ranking'] = model_performance['MAPE'].rank(method='min').astype(int)
"""
# This code saves the model performance results to a CSV file.
model_performance.to_csv(
    f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv", index=False
)

# Define colors for plot
colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Rank models by MAE in ascending order
ranked_models = model_performance.sort_values(by='MAE')

# Abbreviate model names to not more than 6 letters
ranked_models['MODEL_ABBR'] = ranked_models['MODEL'].apply(lambda x: x[:6])

# Initialize figure with two y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot MAE and MAPE for each model
ax1.bar(ranked_models['MODEL_ABBR'], ranked_models['MAE'], color=colors, label='MAE')
ax2.plot(ranked_models['MODEL_ABBR'], ranked_models['MAPE'], color='b', label='MAPE')

# Set title and axis labels
ax1.set_title('Model Performance')
ax1.set_xlabel('Model')
ax1.set_ylabel('MAE')

ax2.set_ylabel('MAPE')

# Set y-axis limits
ax1.set_ylim(0, max(ranked_models['MAE']) * 1.1)
ax2.set_ylim(0, max(ranked_models['MAPE']) * 1.1)

# Add abbreviation labels on top of bars
for i, v in enumerate(ranked_models['MAE']):
    ax1.text(i-0.25, v+0.02, str(round(v,2)), color='black')                                                                  

# Add abbreviation labels on data points
for i, v in enumerate(ranked_models['MAPE']):
    ax2.text(i-0.2, v+0.002, str(round(v,3)), color='b')

# Add legend and sort legend handles by label
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles = handles1 + handles2
labels = labels1 + labels2
ax1.legend(handles, labels, loc='upper right')
handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
ax1.legend(handles, labels, loc='upper right')

# Save plot to file
filename = os.path.join('data', 'model_performance_plot.png')
plt.savefig(filename)

# Close plot
plt.close()
"""

model_performance.to_csv(
    f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv", index=False
)

print(model_performance)