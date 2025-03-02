# -*- coding: utf-8 -*-
"""奥林匹克奖牌预测系统 v1.1
集成自动调参+动态集成+交互可视化+健壮性校验"""

# -*- coding: utf-8 -*-
import sys
import os

# ====== 设置临时目录到F盘 ======
temp_dir = os.path.join('F:/', 'temp_oly')  # F盘创建临时目录
os.environ['TMP'] = temp_dir
os.environ['TEMP'] = temp_dir

# 自动创建目录（如果不存在）
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)  # exist_ok=True避免重复创建报错





import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
from IPython.display import display
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection._split import BaseCrossValidator


# ========== 改进的时间序列分割类 ==========
class BlockingTimeSeriesSplit(BaseCrossValidator):
    """严格防止时间数据泄露的分割方式（兼容scikit-learn接口）"""

    def __init__(self, n_splits=5, min_train_size=3, test_size=1, gap=1):
        super().__init__()
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """生成训练/验证索引（兼容y和groups参数）"""
        # 保持原有逻辑，但确保X是DataFrame且包含'Year'列
        sorted_years = X['Year'].sort_values().unique()
        n_years = len(sorted_years)

        for i in range(self.n_splits):
            train_end_idx = i * (self.test_size + self.gap) + self.min_train_size
            test_end_idx = train_end_idx + self.test_size

            if test_end_idx > n_years:
                break

            train_years = sorted_years[:train_end_idx]
            test_years = sorted_years[train_end_idx + self.gap: test_end_idx + self.gap]

            train_idx = X[X['Year'].isin(train_years)].index.to_numpy()
            test_idx = X[X['Year'].isin(test_years)].index.to_numpy()

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """返回实际可生成的分割数"""
        return min(
            self.n_splits,
            (len(X['Year'].unique()) - self.min_train_size) // (self.test_size + self.gap)
        )

# ========== 数据预处理 ==========
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
    programs = programs.melt(id_vars=['Sport', 'Discipline', 'Code'],
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

    # ========== 改进的项目特征处理 ==========
    # 生成项目特征矩阵
    sport_impact = programs.groupby(['Year', 'Sport'])['EventCount'].sum().unstack().fillna(0)
    sport_impact = sport_impact.add_prefix('Sport_')

    # 合并时保留缺失标记
    medals = pd.merge(medals, sport_impact, on='Year', how='left')

    # 识别新增国家（首次参赛）
    medals['FirstAppearance'] = medals.groupby('NOC')['Year'].transform(
        lambda x: (x == x.min()).astype(int)
    )

    # 分层处理项目缺失值
    sport_cols = [col for col in medals.columns if col.startswith('Sport_')]
    for col in sport_cols:
        # 按国家分组处理
        medals[col] = medals.groupby('NOC')[col].transform(
            lambda x: x.where(
                # 保留已有参赛记录的非零值
                x.notna() | (x.isna() & (medals['FirstAppearance'] == 0)),
                # 新增国家+新项目：填充0
                other=0
            ).ffill(limit=2)  # 允许向前填充最多2届
        )

        # 处理历史国家的新增项目（使用历年该项目的平均参与度）
        medals[col] = medals[col].mask(
            (medals['FirstAppearance'] == 0) & medals[col].isna(),
            medals.groupby('Year')[col].transform('mean')
        )

    # ========== 改进的运动员特征处理 ==========
    # 合并运动员数据
    medals = pd.merge(medals, athlete_features, on=['NOC', 'Year'], how='left')

    # 创建国家参赛标记（基于是否有任何运动员数据）
    athlete_cols = [col for col in medals.columns if col.startswith('Athlete_')]
    medals['HasAthleteData'] = medals[athlete_cols].notna().any(axis=1).astype(int)

    # 分层处理运动员缺失值
    for col in athlete_cols:
        # 时间维度线性插值（国家维度）
        medals[col] = medals.groupby('NOC')[col].transform(
            lambda x: x.interpolate(
                method='linear',
                limit_direction='both',
                limit=1
            )
        )

        # 空间维度处理（基于参赛频率）
        # 计算国家历史参赛频率权重
        country_weight = medals.groupby('NOC')['HasAthleteData'].transform('mean')
        global_mean = medals[col].mean()

        # 使用加权平均值：国家历史参赛率*全局平均值
        medals[col] = medals[col].fillna(
            country_weight * global_mean + (1 - country_weight) * 0
        )

    # ========== 首次获奖标记 ==========
    # 使用插值后的数据重新计算
    medals['Total_adj'] = medals[['Gold', 'Silver', 'Bronze']].sum(axis=1)
    medals['FirstMedal'] = medals.groupby('NOC')['Total_adj'].transform(
        lambda x: (x.cumsum() == x).astype(int)
    )

    return medals.fillna(0)



def validate_features(df, required_cols):
    """特征完整性校验"""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"关键特征缺失: {missing}")
    return df[required_cols].fillna(0)


# ========== 模型核心类 ==========
class AdvancedMedalPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=300,
                                        max_depth=12,
                                        random_state=42,
                                        n_jobs=-1),  # 启用并行
            'xgb': None  # 延迟初始化
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.val_weights = {'rf': 0.6, 'xgb': 0.4}  # 默认权重

    def check_feature_consistency(self, X):
        """特征一致性校验（新增方法）"""
        # 获取预期特征
        expected_features = set(self.feature_names + ['Year'])
        actual_features = set(X.columns)

        # 检查缺失/多余特征
        missing = expected_features - actual_features
        extra = actual_features - expected_features

        if missing:
            raise ValueError(f"预测数据缺失关键特征: {missing}")
        if extra:
            print(f"警告: 预测数据包含多余特征 {extra}, 已自动忽略")

        # 检查Year列类型
        if not pd.api.types.is_integer_dtype(X['Year']):
            raise TypeError("Year列必须是整数类型")




    def optimize_xgb(self, X, y):
        """Optuna超参数优化（集成时间敏感验证）"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0)  # 新增正则化
            }
            model = xgb.XGBRegressor(**params)

            # ==== 关键改进：使用严格时间分割 ====
            tscv = BlockingTimeSeriesSplit(
                n_splits=3,
                min_train_size=4,
                test_size=1,
                gap=1
            )
            score = cross_val_score(model, X, y,
                                    cv=tscv,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=-1).mean()
            return -score

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        return study.best_params

    def train(self, X, y):
        # ==== 关键改进1：强制时间排序 ====
        X = X.sort_values('Year').reset_index(drop=True)
        y = y.loc[X.index]

        # ==== 关键新增：设置特征名称（不包含Year列）====
        self.feature_names = X.drop(columns=['Year']).columns.tolist()

        # 特征校验（校验原始数据包含所有需要的列，包括Year）
        X = validate_features(X, self.feature_names + ['Year'])

        # ==== 关键改进2：标准化前保留年份 ====
        years = X['Year'].copy()
        X_features = X.drop(columns=['Year'])  # 实际用于模型训练的特征
        X_scaled = self.scaler.fit_transform(X_features)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)  # 列名与self.feature_names一致
        X_scaled['Year'] = years.values  # 重新注入时间信息

        # 超参数优化
        print("=== 开始超参数优化 ===")
        best_params = self.optimize_xgb(X_scaled, y)
        self.models['xgb'] = xgb.XGBRegressor(**best_params, n_jobs=-1)
        print(f"优化后的XGB参数: {best_params}")

        # ==== 关键改进3：增强时间序列验证 ====
        tscv = BlockingTimeSeriesSplit(
            n_splits=3,
            min_train_size=4,
            test_size=1,
            gap=1
        )

        fold_weights = []
        for train_idx, val_idx in tscv.split(X_scaled):
            # 严格校验时间顺序
            train_years = X_scaled.iloc[train_idx]['Year']
            val_years = X_scaled.iloc[val_idx]['Year']
            assert val_years.min() > train_years.max(), "时间数据泄露！"

            X_train_fold = X_scaled.iloc[train_idx].drop(columns=['Year'])
            X_val_fold = X_scaled.iloc[val_idx].drop(columns=['Year'])
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # 模型训练
            self.models['rf'].fit(X_train_fold, y_train_fold)
            self.models['xgb'].fit(X_train_fold, y_train_fold)

            # 动态权重计算
            rf_pred = self.models['rf'].predict(X_val_fold)
            xgb_pred = self.models['xgb'].predict(X_val_fold)

            rf_mae = mean_absolute_error(y_val_fold, rf_pred)
            xgb_mae = mean_absolute_error(y_val_fold, xgb_pred)
            total = rf_mae + xgb_mae
            fold_weights.append({
                'rf': xgb_mae / total,
                'xgb': rf_mae / total
            })
            print(f"Fold {len(fold_weights)} 权重: {fold_weights[-1]}")

        # 计算平均权重
        self.val_weights = {
            'rf': np.mean([w['rf'] for w in fold_weights]),
            'xgb': np.mean([w['xgb'] for w in fold_weights])
        }
        print(f"最终集成权重: {self.val_weights}")

        # 全量训练（去除时间列）
        X_final = X_scaled.drop(columns=['Year'])
        self.models['rf'].fit(X_final, y)
        self.models['xgb'].fit(X_final, y)

    def predict_with_uncertainty(self, X, n_bootstraps=500):
        """修复Year列处理错误后的预测方法"""

        # 校验特征
        self.check_feature_consistency(X)
        X = validate_features(X, self.feature_names + ['Year'])

        # 重置索引
        X_features = X.drop(columns=['Year']).reset_index(drop=True)
        years = X['Year'].copy().reset_index(drop=True)

        # 标准化
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )

        # ==== 关键修改：确保 years_values 为一维 ====
        years_values = years.to_numpy().ravel()  # 强制一维化
        years_df = pd.DataFrame({'Year': years_values})

        # 合并时强制重置索引
        X_full = pd.concat([
            X_scaled.reset_index(drop=True),
            years_df.reset_index(drop=True)
        ], axis=1)



        # 重采样
        preds = np.zeros((n_bootstraps, X.shape[0]))
        for i in range(n_bootstraps):
            # 获取唯一年份
            unique_years = X_full['Year'].unique()  # 现在不会报错

            # 分层采样
            sampled_years = np.random.choice(unique_years,
                                             size=len(unique_years),
                                             replace=True)
            sample_mask = X_full['Year'].isin(sampled_years)

            # 预测
            X_sample = X_full[sample_mask].drop(columns=['Year'])
            rf_pred = self.models['rf'].predict(X_sample)
            xgb_pred = self.models['xgb'].predict(X_sample)

            # 存储结果
            preds[i, sample_mask] = (
                    self.val_weights['rf'] * rf_pred +
                    self.val_weights['xgb'] * xgb_pred
            )

        # 处理未采样点
        preds[preds == 0] = np.nan
        mean_pred = np.nanmean(preds, axis=0)
        ci = np.nanpercentile(preds, [5, 95], axis=0)

        return mean_pred, ci


    def explain_features(self, X):
        """时间感知的特征解释"""
        explainer = shap.TreeExplainer(self.models['xgb'])
        X_scaled = self.scaler.transform(X.drop(columns=['Year']))
        shap_values = explainer.shap_values(X_scaled)

        # 将时间信息加入解释
        shap_df = pd.DataFrame({
            'feature': X.drop(columns=['Year']).columns,
            'importance': np.abs(shap_values).mean(0),
            'time_corr': X.drop(columns=['Year']).apply(
                lambda col: col.corr(X['Year']))
        })
        return shap_df.sort_values('importance', ascending=False)

# ========== 可视化系统 ==========
def interactive_prediction_plot(final_report):
    """交互式预测图表"""
    fig = px.scatter(
        final_report,
        x='Predicted_Gold',
        y='NOC',
        error_x='Gold_CI_upper',          # 修改为实际列名
        error_x_minus='Gold_CI_lower',    # 修改为实际列名
        color='Trend',
        title='2028夏季奥运会金牌预测 (含趋势分析)',
        labels={'Predicted_Gold': '预测金牌数'},
        hover_data=['5届平均', 'FirstMedal_Prob']
    )
    fig.update_layout(
        xaxis_range=[0, final_report['Gold_CI_upper'].max() + 10],
        hoverlabel=dict(bgcolor="white", font_size=12),
        height=800
    )
    fig.show()


def plot_sport_correlation(df):
    """项目-奖牌关联分析"""
    sport_cols = [col for col in df.columns if col.startswith('Sport_')]
    corr_matrix = df[sport_cols + ['Gold']].corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(corr_matrix, annot=True, cmap='icefire', center=0, fmt=".2f")
    plt.title('奥运项目与金牌数量相关性分析', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ========== 主程序 ==========
if __name__ == "__main__":
    # 数据加载与预处理（已含改进的缺失值处理）
    medals, hosts, programs, athlete_features = load_and_clean_data()
    df = create_advanced_features(medals, hosts, programs, athlete_features)

    # ==== 动态特征选择改进 ====
    # 确保包含时间敏感特征
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    sport_features = [col for col in numeric_cols if col.startswith('Sport_')]
    time_features = ['Year', 'Gold_3yr_avg', 'Total_trend']  # 新增时间特征
    base_features = ['IsHost']  # 调整基础特征
    candidate_features = base_features + time_features + sport_features[:15]

    # 分层特征筛选
    k_value = max(1, int(len(candidate_features) * 0.7))
    selector = SelectKBest(f_regression, k=k_value)

    X_temp = df.loc[df['Year'] < 2020, candidate_features].fillna(0)
    y_temp = df.loc[df['Year'] < 2020, 'Gold']
    selector.fit(X_temp, y_temp)

    # 确保核心特征（含时间特征）
    selected_features = np.array(candidate_features)[selector.get_support()].tolist()
    essential_features = ['Year', 'Gold_3yr_avg', 'IsHost']  # 强制保留时间特征
    features = list(set(selected_features + essential_features))

    # ==== 数据准备改进 ====
    target = 'Gold'
    train_mask = df['Year'] < 2020

    # 带时间排序的数据准备
    X_train = (df.loc[train_mask, features]
               .sort_values('Year')
               .pipe(validate_features, features))
    y_train = df.loc[train_mask].sort_values('Year')[target]

    # 2028预测数据
    X_future = (df[df['Year'] == 2028][features]
                .pipe(validate_features, features))
    countries_2028 = df[df['Year'] == 2028]['NOC'].values

    # ==== 模型训练改进 ====
    predictor = AdvancedMedalPredictor()
    predictor.feature_names = features
    predictor.train(X_train, y_train)  # 已集成时间敏感训练




    # ==== 预测改进 ====
    gold_pred, gold_ci = predictor.predict_with_uncertainty(X_future)

    # ==== 首次获奖概率预测改进 ====
    from sklearn.utils import resample


    def bootstrap_first_medal(clf, X, n_bootstraps=1000):
        prob_preds = []
        for _ in range(n_bootstraps):
            # 分层重采样（按国家）
            X_resampled, y_resampled = resample(
                X_train,
                df.loc[train_mask, 'FirstMedal'],
                stratify=df.loc[train_mask, 'NOC']
            )
            clf.fit(X_resampled, y_resampled)
            prob_preds.append(clf.predict_proba(X)[:, 1])
        return np.mean(prob_preds, axis=0), np.percentile(prob_preds, [2.5, 97.5], axis=0)


    first_medal_features = [col for col in features if col != 'FirstMedal']
    clf = CalibratedClassifierCV(
        xgb.XGBClassifier(scale_pos_weight=7, eval_metric='logloss'),
        method='isotonic',
        cv=BlockingTimeSeriesSplit(n_splits=3)  # 时间敏感交叉验证
    )
    clf.fit(X_train[first_medal_features], df.loc[train_mask, 'FirstMedal'])

    first_probs, first_ci = bootstrap_first_medal(
        clf,
        X_future[first_medal_features]
    )

    # ==== 最终报告生成 ====
    final_report = pd.DataFrame({
        'NOC': countries_2028,
        'Predicted_Gold': np.round(gold_pred, 1),
        'Gold_CI_lower': np.round(gold_ci[0], 1),
        'Gold_CI_upper': np.round(gold_ci[1], 1),
        'FirstMedal_Prob': np.round(first_probs, 3),
        'First_CI_lower': np.round(first_ci[0], 3),
        'First_CI_upper': np.round(first_ci[1], 3)
    })

    # 趋势分析改进（使用滑动窗口对比）
    history_window = df[(df['Year'] >= 2012) & (df['Year'] <= 2024)]
    avg_gold = history_window.groupby('NOC')['Gold'].mean().rename('5届平均')
    final_report = final_report.merge(
        avg_gold,
        left_on='NOC',
        right_index=True,
        how='left'
    )
    final_report['Trend'] = np.select(
        [
            final_report['Predicted_Gold'] > final_report['5届平均'] * 1.2,
            final_report['Predicted_Gold'] < final_report['5届平均'] * 0.8
        ],
        ['↑↑ 显著上升', '↓↓ 显著下降'],
        default='→ 平稳'
    )

    # ==== 可视化改进 ====
    print("\n=== 2028奥运会奖牌预测最终报告（带双重置信区间）===")
    display(final_report.sort_values('Predicted_Gold', ascending=False).head(15))

    print("\n=== 时间敏感特征分析 ===")
    time_importance = predictor.explain_features(X_train)
    print(time_importance[['feature', 'time_corr']].sort_values('time_corr', ascending=False))

    print("\n=== 交互式双重置信区间可视化 ===")
    interactive_prediction_plot(final_report)

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    csv_filename = "2028_奥运会奖牌预测报告.csv"
    csv_path = os.path.join(desktop_path, csv_filename)
    final_report.to_csv(csv_path, index=False, encoding="utf_8_sig")
    print(f"\n=== CSV报告已生成 ===\n保存路径：{csv_path}")