# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:20:22 2024
奥林匹克奖牌预测系统 v1.0
@author: hhh
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import chardet
from pylab import mpl
from scipy.stats import mstats

# ========== Global Configurations ==========
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# ========== Function Definitions ==========

def detect_encoding(file_path):
    """Automatically detect file encoding"""
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(100000))['encoding']

def read_csv_with_encoding(file_path):
    """Read CSV with automatic encoding handling"""
    encodings = ['utf-8', 'latin1', 'gbk', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"Successfully read {file_path}, encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
    detected_enc = detect_encoding(file_path)
    return pd.read_csv(file_path, encoding=detected_enc)

# ========== Data Loading ==========
# Load medal data
df_medals = read_csv_with_encoding('summerOly_medal_counts.csv')
df_hosts = read_csv_with_encoding('summerOly_hosts.csv')
df_programs = read_csv_with_encoding('summerOly_programs.csv')
df_athletes = read_csv_with_encoding('summerOly_athletes.csv')

# ========== Data Preprocessing ==========
# 1. Clean host country data
df_hosts['HostCountry'] = df_hosts['Host'].str.split(',').str[-1].str.strip()

country_noc_map = {
    'Greece': 'GRE', 'France': 'FRA', 'United States': 'USA',
    'United Kingdom': 'GBR', 'Sweden': 'SWE', 'Belgium': 'BEL',
    'Netherlands': 'NED', 'Germany': 'GER', 'Finland': 'FIN',
    'Australia': 'AUS', 'Italy': 'ITA', 'Spain': 'ESP',
    'Soviet Union': 'URS', 'West Germany': 'FRG', 'East Germany': 'GDR',
    'Unified Team': 'EUN', 'ROC': 'ROC', 'North Korea': 'PRK',
    'South Korea': 'KOR', 'Mixed team': 'MIX', 'Great Britain': 'GBR',
    'Iceland': 'ISL', 'Ghana': 'GHA', 'Iraq': 'IRQ', 'Malaysia': 'MAS',
    'Kuwait': 'KUW', 'Paraguay': 'PAR', 'Sudan': 'SUD', 'Saudi Arabia': 'KSA'
}

df_hosts['HostNOC'] = df_hosts['HostCountry'].map(country_noc_map).fillna('UNK')

# 2. Merge host country info into medal data
df_medals = pd.merge(df_medals, df_hosts[['Year','HostNOC']], on='Year', how='left')
df_medals['Host'] = (df_medals['NOC'] == df_medals['HostNOC']).astype(int)

# 3. Calculate historical gold averages
df_historical = df_medals.groupby('NOC')['Gold'].mean().reset_index(name='HistoricalGold')
df_medals = pd.merge(df_medals, df_historical, on='NOC', how='left')

# 4. Process sport features
sport_features = df_athletes.pivot_table(
    index=['NOC','Year'],
    columns='Sport',
    values='Event',
    aggfunc='count',
    fill_value=0
)
sport_columns = list(sport_features.columns)  # Get all sport column names
df_medals = pd.merge(df_medals, sport_features.reset_index(), on=['NOC','Year'], how='left').fillna(0)

# 5. Time series features (critical corrections)
def safe_pct_change(x):
    shifted = x.shift(1)
    denominator = shifted.replace(0, 1e-6)  # Prevent division by zero
    growth = (x - shifted) / denominator
    return growth.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN

df_medals['Gold_3yr_avg'] = df_medals.groupby('NOC')['Gold'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df_medals['Gold_growth'] = df_medals.groupby('NOC')['Gold'].transform(safe_pct_change)

# Handle missing values and outliers (fix chained assignment warning)
df_medals = df_medals.assign(Gold_growth=df_medals['Gold_growth'].fillna(0))
lower = df_medals['Gold_growth'].quantile(0.01)
upper = df_medals['Gold_growth'].quantile(0.99)
df_medals['Gold_growth'] = df_medals['Gold_growth'].clip(lower, upper)

# 6. Data standardization
num_cols = ['HistoricalGold', 'Gold_3yr_avg', 'Gold_growth']
for col in num_cols:
    df_medals[col] = mstats.winsorize(df_medals[col], limits=[0.01, 0.01])

scaler = StandardScaler()
df_medals.loc[:, num_cols] = scaler.fit_transform(df_medals[num_cols])

# ========== Model Training ==========
# Define features and targets
features = ['Host', 'HistoricalGold', 'Gold_3yr_avg', 'Gold_growth'] + sport_columns
target_gold = 'Gold'
target_total = 'Total'

# Split data into train/test sets for gold medal prediction
X_train, X_test, y_train_gold, y_test_gold = train_test_split(
    df_medals[features], df_medals[target_gold], test_size=0.2, random_state=42
)

# Split data into train/test sets for total medals prediction
y_train_total, y_test_total = train_test_split(df_medals[target_total], test_size=0.2, random_state=42)

# 1. Random Forest model for gold medals
rf_gold = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf_gold.fit(X_train, y_train_gold)

# 2. XGBoost model for gold medals
xgb_gold = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)
xgb_gold.fit(X_train, y_train_gold)

# Ensemble prediction for gold medals
gold_pred = 0.6 * rf_gold.predict(X_test) + 0.4 * xgb_gold.predict(X_test)

# Evaluation metrics for gold medals
print(f"MAE (Gold): {mean_absolute_error(y_test_gold, gold_pred):.2f}")
print(f"R² (Gold): {r2_score(y_test_gold, gold_pred):.2f}")

# ========== Model Training for Total Medals ==========
# 1. Random Forest model for total medals
rf_total = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf_total.fit(X_train, y_train_total)

# 2. XGBoost model for total medals
xgb_total = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)
xgb_total.fit(X_train, y_train_total)

# Ensemble prediction for total medals
total_pred = 0.6 * rf_total.predict(X_test) + 0.4 * xgb_total.predict(X_test)

# Evaluation metrics for total medals
print(f"MAE (Total): {mean_absolute_error(y_test_total, total_pred):.2f}")
print(f"R² (Total): {r2_score(y_test_total, total_pred):.2f}")

# ========== Visualization ==========
# 1. Gold Medal Feature Importance
plt.figure(figsize=(12,6))
plt.barh(features[:10], rf_gold.feature_importances_[:10])
plt.title('Top 10 Feature Importance for Gold Medal Prediction')
plt.show()

# 2. Gold Medal Prediction Interval Visualization
plt.figure(figsize=(14,7))
plt.plot(y_test_gold.values, label='Actual Gold Medals')
plt.plot(gold_pred, label='Predicted Gold Medals')
plt.fill_between(
    range(len(y_test_gold)),
    gold_pred - 5,
    gold_pred + 5,
    alpha=0.3,
    label='Prediction Interval'
)
plt.legend()
plt.title('Gold Medal Count Predictions')
plt.show()

# 3. Total Medal Prediction Interval Visualization
plt.figure(figsize=(14,7))
plt.plot(y_test_total.values, label='Actual Total Medals')
plt.plot(total_pred, label='Predicted Total Medals')
plt.fill_between(
    range(len(y_test_total)),
    total_pred - 5,
    total_pred + 5,
    alpha=0.3,
    label='Prediction Interval'
)
plt.legend()
plt.title('Total Medal Count Predictions')
plt.show()

# ========== 2028 Prediction ==========
# Construct prediction data for 2028
data_2028 = pd.DataFrame({
    'Host': [1, 0, 0],
    'HistoricalGold': [45, 38, 27],
    'Gold_3yr_avg': [42, 36, 25],
    'Gold_growth': [0.05, 0.03, -0.02]
})

# Initialize all sport columns to 0
for col in sport_columns:
    data_2028[col] = 0

# Set known sport participation counts
data_2028['Athletics'] = [15, 12, 10]
data_2028['Swimming'] = [12, 10, 8]
data_2028['Gymnastics'] = [8, 6, 5]

# Standardize numerical features
data_2028[num_cols] = scaler.transform(data_2028[num_cols])

# Ensure feature order consistency
X_2028 = data_2028[features]

# Gold medal prediction for 2028
gold_2028 = 0.6 * rf_gold.predict(X_2028) + 0.4 * xgb_gold.predict(X_2028)
print(f"Predicted gold medals for 2028: {gold_2028}")

# Total medal prediction for 2028
total_2028 = 0.6 * rf_total.predict(X_2028) + 0.4 * xgb_total.predict(X_2028)
print(f"Predicted total medals for 2028: {total_2028}")
