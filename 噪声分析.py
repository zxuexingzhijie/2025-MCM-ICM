# -*- coding: utf-8 -*-
"""
Olympic Medal Prediction and Robustness Analysis Complete Code
Includes Gaussian Noise Sensitivity Testing and Visualization
"""

# ========== Basic Library Imports ==========
import pandas as pd
import numpy as np
from scipy.stats import mstats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import chardet
from pylab import mpl
from scipy import stats
import warnings

# ========== Global Configurations ==========
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Chinese display
mpl.rcParams['axes.unicode_minus'] = False  # Negative sign display
warnings.filterwarnings('ignore')  # Filter warnings
np.random.seed(42)  # Random seed


# ========== Custom Functions ==========
def detect_encoding(file_path):
    """Automatically detect file encoding"""
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(100000))['encoding']


def read_csv_with_encoding(file_path):
    """CSV reader with automatic encoding handling"""
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


def add_gaussian_noise(data, noise_std=0.1, exclude_cols=['Host']):
    """Add Gaussian noise to continuous features"""
    noisy_data = data.copy()
    numeric_cols = [col for col in data.columns if col not in exclude_cols]
    noise = np.random.normal(loc=0, scale=noise_std, size=data[numeric_cols].shape)
    noisy_data[numeric_cols] += noise
    return noisy_data


# ========== Data Loading & Preprocessing ==========
print("\n====== Data Loading ======")
# Load datasets
df_medals = read_csv_with_encoding('summerOly_medal_counts.csv')
df_hosts = read_csv_with_encoding('summerOly_hosts.csv')
df_athletes = read_csv_with_encoding('summerOly_athletes.csv')

# Host country processing
print("\n====== Feature Engineering ======")
df_hosts['HostCountry'] = df_hosts['Host'].str.split(',').str[-1].str.strip()
country_noc_map = {'United States': 'USA', 'Australia': 'AUS', 'Japan': 'JPN',
                   'France': 'FRA', 'Great Britain': 'GBR', 'China': 'CHN'}
df_hosts['HostNOC'] = df_hosts['HostCountry'].map(country_noc_map).fillna('UNK')

# Merge host information
df_medals = pd.merge(df_medals, df_hosts[['Year', 'HostNOC']], on='Year', how='left')
df_medals['Host'] = (df_medals['NOC'] == df_medals['HostNOC']).astype(int)

# Sport participation features
sport_features = df_athletes.pivot_table(
    index=['NOC', 'Year'],
    columns='Sport',
    values='Event',
    aggfunc='count',
    fill_value=0
)
sport_columns = list(sport_features.columns)
df_medals = pd.merge(df_medals, sport_features.reset_index(), on=['NOC', 'Year'], how='left').fillna(0)


# Time-series features
def safe_pct_change(x):
    shifted = x.shift(1).replace(0, 1e-6)
    return ((x - shifted) / shifted).replace([np.inf, -np.inf], np.nan)


df_medals['Gold_3yr_avg'] = df_medals.groupby('NOC')['Gold'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df_medals['Gold_growth'] = df_medals.groupby('NOC')['Gold'].transform(safe_pct_change).fillna(0)

# Outlier handling
num_cols = ['Gold_3yr_avg', 'Gold_growth']
for col in num_cols:
    df_medals[col] = mstats.winsorize(df_medals[col], limits=[0.01, 0.01])

# Standardization
scaler = StandardScaler()
df_medals[num_cols] = scaler.fit_transform(df_medals[num_cols])

# ========== Model Training ==========
print("\n====== Model Training ======")
features = ['Host', 'Gold_3yr_avg', 'Gold_growth'] + sport_columns
X_train, X_test, y_train, y_test = train_test_split(
    df_medals[features], df_medals['Gold'], test_size=0.2, random_state=42
)

# Initialize models
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6
)

# Train models
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Ensemble prediction
gold_pred = 0.6 * rf.predict(X_test) + 0.4 * xgb_model.predict(X_test)
print(f"\nBaseline Model Performance:\nMAE: {mean_absolute_error(y_test, gold_pred):.2f}\nR²: {r2_score(y_test, gold_pred):.2f}")

# ========== Robustness Analysis ==========
print("\n====== Robustness Analysis ======")
noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
n_iterations = 30
baseline_mae = mean_absolute_error(y_test, gold_pred)

# Result storage
results = {'MAE': {level: [] for level in noise_levels},
           'R²': {level: [] for level in noise_levels}}

for level in noise_levels:
    print(f"Processing noise level {level}...")
    for _ in range(n_iterations):
        # Add noise and retrain
        X_train_noisy = add_gaussian_noise(X_train, level)

        rf_temp = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
        xgb_temp = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6)

        rf_temp.fit(X_train_noisy, y_train)
        xgb_temp.fit(X_train_noisy, y_train)

        pred = 0.6 * rf_temp.predict(X_test) + 0.4 * xgb_temp.predict(X_test)
        results['MAE'][level].append(mean_absolute_error(y_test, pred))
        results['R²'][level].append(r2_score(y_test, pred))

# ========== Visualization ==========
print("\n====== Generating Visualizations ======")
plt.figure(figsize=(15, 10))

# MAE distribution
plt.subplot(2, 2, 1)
for level in noise_levels:
    plt.scatter([level] * n_iterations, results['MAE'][level], alpha=0.6,
                label=f'σ={level}' if level == 0.05 else '')
plt.axhline(baseline_mae, color='r', linestyle='--', label='Baseline MAE')
plt.title('MAE Trend with Noise Level')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('MAE')
plt.legend()

# R² distribution
plt.subplot(2, 2, 2)
for level in noise_levels:
    plt.scatter([level] * n_iterations, results['R²'][level], alpha=0.6)
plt.axhline(r2_score(y_test, gold_pred), color='r', linestyle='--', label='Baseline R²')
plt.title('R² Trend with Noise Level')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('R²')

# Statistical significance
plt.subplot(2, 2, 3)
p_values = [stats.ttest_1samp(results['MAE'][level], baseline_mae).pvalue
            for level in noise_levels]
plt.semilogy(noise_levels, p_values, 'bo-')
plt.axhline(0.05, color='r', linestyle='--')
plt.title('Statistical Significance Analysis (t-test)')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('p-value (log scale)')

# Sensitivity heatmap
plt.subplot(2, 2, 4)
mae_changes = [np.mean(results['MAE'][level]) / baseline_mae - 1 for level in noise_levels]
plt.imshow([mae_changes], cmap='Reds', aspect=0.2,
           extent=[noise_levels[0], noise_levels[-1], 0, 1])
plt.colorbar(label='Relative MAE Change')
plt.title('Noise Sensitivity Heatmap')
plt.yticks([])
plt.xlabel('Noise Standard Deviation')

plt.tight_layout()
plt.show()

# ========== 2028 Prediction ==========
print("\n====== 2028 Prediction ======")
data_2028 = pd.DataFrame({
    'Host': [1, 0, 0],
    'Gold_3yr_avg': [42, 36, 25],
    'Gold_growth': [0.05, 0.03, -0.02]
})

for col in sport_columns:
    data_2028[col] = 0
data_2028[['Athletics', 'Swimming', 'Gymnastics']] = [[15, 12, 8], [12, 10, 6], [10, 8, 5]]

data_2028[num_cols] = scaler.transform(data_2028[num_cols])
X_2028 = data_2028[features]

gold_2028 = 0.6 * rf.predict(X_2028) + 0.4 * xgb_model.predict(X_2028)
print(f"2028 Gold Medal Predictions: {gold_2028.round(1)}")