# -*- coding: utf-8 -*-
"""
奥林匹克奖牌预测系统 v2.2
集成RF+XGBoost模型、首次获奖预测和完整可视化
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# ========== 数据加载与预处理 ==========
def load_and_clean_data():



    medals = pd.read_csv('summerOly_medal_counts.csv')
    hosts = pd.read_csv('summerOly_hosts.csv')
    programs = pd.read_csv('summerOly_programs.csv')
    athletes = pd.read_csv('summerOly_athletes.csv')


    # 统一国家代码
    country_map = {
        'Greece': 'GRE',
        'France': 'FRA',
        'United States': 'USA',
        'United Kingdom': 'GBR',
        'Sweden': 'SWE',
        'Belgium': 'BEL',
        'Netherlands': 'NED',
        'Germany': 'GER',
        'Finland': 'FIN',
        'Australia': 'AUS',
        'Italy': 'ITA',
        'Spain': 'ESP',
        'Soviet Union': 'URS',
        'West Germany': 'FRG',
        'East Germany': 'GDR',
        'Unified Team': 'EUN',
        'ROC': 'ROC',  # Russian Olympic Committee
        'North Korea': 'PRK',
        'South Korea': 'KOR',
        'Mixed team': 'MIX',
        'Great Britain': 'GBR',
        'Iceland': 'ISL',
        'Ghana': 'GHA',
        'Iraq': 'IRQ',
        'Malaysia': 'MAS',
        'Kuwait': 'KUW',
        'Paraguay': 'PAR',
        'Sudan': 'SUD',
        'Saudi Arabia': 'KSA',
    }
    medals['NOC'] = medals['NOC'].replace(country_map)

    # 处理主办国数据
    hosts[['City', 'Country']] = hosts['Host'].str.extract(r'([^,]+),\s*(.+)')
    hosts['Country'] = hosts['Country'].replace({'West Germany': 'GER', 'East Germany': 'GER'})

    # 项目数据处理
    programs = programs.melt(id_vars=['Sport', 'Discipline','Code'],
                             var_name='Year', value_name='Events')

    # 仅选择有效的年份列
    year_columns = [col for col in programs.columns if col.isdigit()]
    # 修改后：使用唯一列名
    programs = programs.melt(
        id_vars=['Sport', 'Discipline', 'Code'],
        value_vars=year_columns,
        var_name='Year',
        value_name='EventCount'  # 修改为唯一名称
    )

    # 安全转换年份并过滤无效值
    programs['Year'] = programs['Year'].apply(
        lambda x: int(x) if x.isdigit() else None
    )
    programs = programs.dropna(subset=['Year'])



    # 运动员数据特征
    athletes['Medal'] = athletes['Medal'].map({'Gold': 3, 'Silver': 2, 'Bronze': 1, 'No medal': 0})
    athlete_features = athletes.groupby(['NOC', 'Year', 'Sport'])['Medal'].max().unstack().add_prefix('Athlete_')

    # 确保插入所有国家的2028年数据
    all_nocs = medals['NOC'].unique()
    placeholder_2028 = pd.DataFrame({
        'NOC': all_nocs,
        'Year': 2028,
        'Gold': 0, 'Silver': 0, 'Bronze': 0, 'Total': 0
    })
    medals = pd.concat([medals, placeholder_2028], ignore_index=True)

    # 确保插入主办国信息
    if 2028 not in hosts['Year'].values:
        hosts = pd.concat([
            hosts,
            pd.DataFrame({'Year': [2028], 'Host': 'Los Angeles, United States'})
        ], ignore_index=True)



    return medals, hosts, programs, athlete_features


# ========== 高级特征工程 ==========
def create_advanced_features(medals, hosts, programs, athlete_features):


    # 时间序列特征
    medals['Gold_3yr_avg'] = medals.groupby('NOC')['Gold'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    medals['Total_trend'] = medals.groupby('NOC')['Total'].transform(
        lambda x: x.pct_change(periods=2).fillna(0))

    # 主办国特征
    host_data = hosts[['Year', 'Country']].rename(columns={'Country': 'HostCountry'})
    medals = pd.merge(medals, host_data, on='Year', how='left')
    medals['IsHost'] = (medals['NOC'] == medals['HostCountry']).astype(int)

    # 项目特征合并时填充缺失值为0
    sport_impact = programs.groupby(['Year', 'Sport'])['EventCount'].sum().unstack().fillna(0)
    sport_impact = sport_impact.add_prefix('Sport_')
    medals = pd.merge(medals, sport_impact, on='Year', how='left').fillna(0)


    # 运动员特征
    medals = pd.merge(medals, athlete_features, on=['NOC', 'Year'], how='left')

    # 首次获奖标记
    medals['FirstMedal'] = medals.groupby('NOC')['Total'].transform(
        lambda x: (x.cumsum() == x).astype(int))

    return medals.fillna(0)


# ========== 模型构建与预测 ==========
class AdvancedMedalPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
            'xgb': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, learning_rate=0.1)
        }
        self.scaler = StandardScaler()
        self.feature_names = []

    def train(self, X, y):
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        print("=== Model Performance ===")
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                     scoring='neg_mean_absolute_error')
            print(f"{name} MAE: {-scores.mean():.2f} (±{scores.std():.2f})")
            model.fit(X_scaled, y)

    def predict_with_uncertainty(self, X, n_bootstraps=500):
        # 添加特征对齐校验
        X = X[self.feature_names]  # 按训练时的特征顺序排列
        X_scaled = self.scaler.transform(X)
        preds = np.zeros((n_bootstraps, X.shape[0]))




        # Bootstrap集成预测
        for i in range(n_bootstraps):
            sample_idx = resample(np.arange(X.shape[0]))
            rf_pred = self.models['rf'].predict(X_scaled[sample_idx])
            xgb_pred = self.models['xgb'].predict(X_scaled[sample_idx])
            preds[i] = 0.6 * rf_pred + 0.4 * xgb_pred  # 加权集成

        mean_pred = np.mean(preds, axis=0)
        ci = np.percentile(preds, [5, 95], axis=0)
        return mean_pred, ci

    def explain_features(self):
        # SHAP解释
        explainer = shap.TreeExplainer(self.models['xgb'])
        shap_values = explainer.shap_values(self.scaler.transform(X_train))
        return shap_values


# ========== 可视化系统 ==========
def plot_2028_predictions(predictions, ci, countries):
    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(countries))
    plt.errorbar(predictions, y_pos, xerr=[predictions - ci[0], ci[1] - predictions],
                 fmt='o', color='#2c7bb6', ecolor='#d7191c', capsize=5,
                 markersize=8, elinewidth=2)
    plt.yticks(y_pos, countries)
    plt.title('2028 Summer Olympics Gold Medal Predictions with 90% CI', fontsize=14)
    plt.xlabel('Predicted Gold Medals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_first_medal_probabilities(prob_df):
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='FirstProb',
        y='NOC',
        data=prob_df,
        hue='NOC',  # 添加hue参数
        palette='viridis',
        legend=False  # 关闭图例
    )
    plt.title('First Medal Winning Probabilities for 2028 Olympics', fontsize=14)
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.xlim(0, 1)
    plt.axvline(x=0.5, color='red', linestyle='--')
    plt.text(0.52, 0.9 * len(prob_df), '50% Probability Threshold',
             color='red', transform=plt.gca().transData)
    plt.tight_layout()
    plt.show()





# ========== 主程序 ==========
if __name__ == "__main__":
    # 数据准备
    medals, hosts, programs, athlete_features = load_and_clean_data()
    df = create_advanced_features(medals, hosts, programs, athlete_features)

    # ==== 修改1：动态特征选择 ====
    from sklearn.feature_selection import SelectKBest, f_regression  # 新增导入
    import numpy as np

    # 生成候选特征列表（仅数值型）
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    sport_features = [col for col in numeric_cols if col.startswith('Sport_')]
    athlete_features = [col for col in numeric_cols if col.startswith('Athlete_')]
    base_features = ['Gold_3yr_avg', 'Total_trend', 'IsHost']
    candidate_features = base_features + sport_features + athlete_features

    # 特征重要性筛选（保留前15重要特征）
    selector = SelectKBest(f_regression, k=15)
    X_temp = df.loc[df['Year'] < 2020, candidate_features].fillna(0)  # 训练集数据
    y_temp = df.loc[df['Year'] < 2020, 'Gold']
    selector.fit(X_temp, y_temp)

    # 强制保留核心特征
    essential_features = ['Gold_3yr_avg', 'IsHost']  # 必须保留的时序特征
    selected_features = np.array(candidate_features)[selector.get_support()].tolist()
    features = list(set(selected_features + essential_features))  # 合并特征

    # 检查特征是否存在
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"关键特征缺失: {missing_features}")

    # ==== 修改2：数据预处理 ====
    target = 'Gold'
    train_mask = df['Year'] < 2020
    X_train = df.loc[train_mask, features].fillna(0)  # 显式处理缺失值
    y_train = df.loc[train_mask, target]
    X_future = df[df['Year'] == 2028][features].fillna(0)
    countries_2028 = df[df['Year'] == 2028]['NOC'].values

    # 训练模型
    predictor = AdvancedMedalPredictor()
    predictor.train(X_train, y_train)

    # 生成预测
    gold_pred, gold_ci = predictor.predict_with_uncertainty(X_future)

    # 可视化预测结果
    plot_2028_predictions(gold_pred, gold_ci, countries_2028)

    # ==== 修改3：首次获奖预测（概率校准）====
    from sklearn.calibration import CalibratedClassifierCV  # 新增导入

    # ==== 首次获奖预测（独立特征集）====
    # 定义特征（排除FirstMedal）
    first_medal_features = [col for col in features if col != 'FirstMedal']

    # 准备数据
    X_first = df[first_medal_features].fillna(0)
    y_first = df['FirstMedal']

    # 训练校准模型
    base_clf = xgb.XGBClassifier(
        scale_pos_weight=7,
        eval_metric='logloss'
    )
    calibrated_model = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    calibrated_model.fit(X_first, y_first)

    # 生成预测概率（使用相同特征）
    future_probs = pd.DataFrame({
        'NOC': countries_2028,
        'FirstProb': calibrated_model.predict_proba(X_future[first_medal_features])[:, 1]
    })


    future_probs = future_probs[future_probs['FirstProb'] > 0.1].sort_values('FirstProb', ascending=False)

    # 可视化首次获奖概率
    plot_first_medal_probabilities(future_probs.head(10))

    # 修改最终报告生成逻辑
    final_report = pd.DataFrame({
        'NOC': future_probs['NOC'].values,  # 使用筛选后的国家列表
        'Predicted_Gold': np.round(gold_pred[future_probs.index], 1),  # 按筛选索引对齐
        'CI_lower': np.round(gold_ci[0][future_probs.index], 1),
        'CI_upper': np.round(gold_ci[1][future_probs.index], 1),
        'FirstMedal_Probability': np.round(future_probs['FirstProb'].values, 3)
    }).sort_values('Predicted_Gold', ascending=False)

    # 添加趋势分析
    latest_gold = df[df['Year'] == 2024].set_index('NOC')['Gold']
    final_report['2024_Gold'] = final_report['NOC'].map(latest_gold)
    final_report['Trend'] = np.where(
        final_report['Predicted_Gold'] > final_report['2024_Gold'],
        '↑ Improving',
        '↓ Declining'
    )

    print("\n=== 2028 Olympics Final Projection ===")
    print(final_report.head(15))

    # SHAP特征解释
    print("\n=== Feature Importance Analysis ===")
    shap_values = predictor.explain_features()
    shap.summary_plot(shap_values, X_train, plot_type='bar')