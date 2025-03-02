import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据，指定编码为 ISO-8859-1
athletes = pd.read_csv('summerOly_athletes.csv', encoding='ISO-8859-1')
medal_counts = pd.read_csv('summerOly_medal_counts.csv', encoding='ISO-8859-1')

# 数据预处理
# 1. 奖牌数据：按国家和年份汇总
medal_summary = medal_counts.groupby(['Year', 'NOC']).agg(
    {'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum', 'Total': 'sum'}).reset_index()

# 2. 运动员数据：按国家和年份汇总参赛人数
athlete_summary = athletes.groupby(['Year', 'NOC']).size().reset_index(name='Athletes')

# 3. 合并数据
data = pd.merge(medal_summary, athlete_summary, on=['Year', 'NOC'], how='left')
data = data.fillna(0)  # 填充缺失值

# 4. 添加历史奖牌特征
# 计算每个国家过去3届奥运会的平均奖牌数和总奖牌数
data['Past_Gold_Avg'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Total_Avg'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).mean())
data['Past_Gold_Sum'] = data.groupby('NOC')['Gold'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())
data['Past_Total_Sum'] = data.groupby('NOC')['Total'].transform(
    lambda x: x.shift().rolling(window=3, min_periods=1).sum())

# 填充历史特征的缺失值（对于早期数据）
data[['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']] = data[[
    'Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']].fillna(0)

# 特征与目标
features = ['Past_Gold_Avg', 'Past_Total_Avg', 'Past_Gold_Sum', 'Past_Total_Sum']
target_gold = 'Gold'
target_total = 'Total'


# 训练 Ridge 回归模型
def build_ridge_model(data, features, target):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)  # 使用Ridge回归，alpha是正则化强度
    model.fit(X_train, y_train)

    # 预测并计算MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 计算R²
    r2 = model.score(X_test, y_test)
    print(f'Model for {target}: MSE = {mse}, R² = {r2}')

    return model, X_test, y_test, y_pred, r2


# 构建金牌和总奖牌的模型
ridge_gold, X_test_gold, y_test_gold, gold_pred, r2_gold = build_ridge_model(data, features, target_gold)
ridge_total, X_test_total, y_test_total, total_pred, r2_total = build_ridge_model(data, features, target_total)

# 打印R²
print(f'R² for Gold Medal Prediction: {r2_gold}')
print(f'R² for Total Medal Prediction: {r2_total}')

# ========== Visualization ==========

# 1. Gold Medal Feature Importance
# Ridge模型没有直接的特征重要性，因此我们展示回归系数作为特征的重要性
plt.figure(figsize=(12, 6))
plt.barh(features, ridge_gold.coef_)
plt.title('Feature Coefficients for Gold Medal Prediction')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

# 2. Gold Medal Prediction Interval Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test_gold.values, label='Actual Gold Medals', color='blue')
plt.plot(gold_pred, label='Predicted Gold Medals', color='red')
plt.fill_between(
    range(len(y_test_gold)),
    gold_pred - 5,
    gold_pred + 5,
    alpha=0.3,
    label='Prediction Interval'
)
plt.legend()
plt.title('Gold Medal Count Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Gold Medal Count')
plt.show()

# 3. Total Medal Prediction Interval Visualization
plt.figure(figsize=(14, 7))
plt.plot(y_test_total.values, label='Actual Total Medals', color='green')
plt.plot(total_pred, label='Predicted Total Medals', color='orange')
plt.fill_between(
    range(len(y_test_total)),
    total_pred - 5,
    total_pred + 5,
    alpha=0.3,
    label='Prediction Interval'
)
plt.legend()
plt.title('Total Medal Count Predictions')
plt.xlabel('Sample Index')
plt.ylabel('Total Medal Count')
plt.show()
