import pandas as pd

# 加载奖牌数据
medal_data_path = "summerOly_medal_counts.csv"
medal_data = pd.read_csv(medal_data_path)

# 加载主办国家数据
host_data_path = 'summerOly_hosts.csv'
host_data = pd.read_csv(host_data_path)

# 处理主办国家字段
host_data['Host Country'] = host_data['Host'].apply(lambda x: x.split(',')[-1].strip())

# 合并数据，通过年份将奖牌数据与主办国家数据关联
combined_data = pd.merge(medal_data, host_data, on="Year", how="inner")

# 计算是否为主办国家
combined_data['Is Host'] = combined_data['NOC'] == combined_data['Host Country']
host_medals = combined_data[combined_data['Is Host']]

# 计算主办国奖牌总数与总奖牌数的相关性
correlation_host_medals = host_medals['Total'].corr(combined_data['Total'])

# 输出相关性结果
print("Correlation between hosting the Olympics and total medals won:", correlation_host_medals)

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))

# 使用seaborn样式
sns.set_style("whitegrid")
sns.set_palette("pastel")

# 绘制散点图（调整点的大小和样式）
ax = sns.scatterplot(
    data=host_medals,
    x="Year",
    y="Total",
    s=150,  # 增大点的大小
    color="#FFA500",  # 使用更醒目的橙色
    edgecolor='black',  # 添加黑色边框
    linewidth=1.5,  # 边框粗细
    zorder=2  # 确保点在上层
)

# 设置标题和标签样式
plt.title('Host Nation Medal Performance in Summer Olympics\n1896-2024',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Year", fontsize=12, labelpad=10)
plt.ylabel("Total Medals Won", fontsize=12, labelpad=10)

# 调整坐标轴
plt.xticks(host_medals["Year"].unique(), rotation=45, fontsize=10)
plt.yticks(fontsize=10)
ax.set_xlim(host_medals["Year"].min()-4, host_medals["Year"].max()+4)

# 添加数据标签
for index, row in host_medals.iterrows():
    ax.text(
        x=row["Year"]+0.8,  # 水平偏移
        y=row["Total"]+5,   # 垂直偏移
        s=f'{int(row["Total"])}',
        fontsize=9,
        ha='left',
        va='bottom'
    )

# 添加参考线
ax.axhline(y=host_medals['Total'].median(),
           color='gray',
           linestyle='--',
           linewidth=1,
           alpha=0.7,
           zorder=1)

# 优化网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 移除多余的边框
sns.despine(left=True, bottom=True)



# 调整布局
plt.tight_layout()

plt.show()