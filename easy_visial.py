import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# 设置主题和颜色
sns.set_theme(style="whitegrid")
palette = ["#fed892"]

# 设置字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('charts/sentiments/smart_home_ques-sentiments.csv')

# 1. 趋势预测
# 以发布时间为x轴，回答数量为y轴，绘制时间序列图
df['发布时间'] = pd.to_datetime(df['发布时间'])
df.set_index('发布时间', inplace=True)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df.resample('M').size(), palette=palette)
plt.title('Trend Over Time')
plt.ylabel('Number of Answers')
plt.savefig(r"charts\trends\Trend-Over-Time.png")
plt.show()

# 2. 用户活跃度分析
# 以用户ID为x轴，回答数量为y轴，绘制条形图
plt.figure(figsize=(10, 6))
active_users = df.groupby('用户名').size().sort_values(ascending=False).head(10)
sns.barplot(x=active_users.index, y=active_users.values, palette=palette)
plt.title('Top 10 Active Users')
plt.ylabel('Number of Answers')
plt.xticks(rotation=45)
plt.savefig(r"charts\trends\Top-10-Active-Users.png")
plt.show()

# 3. 用户影响力分析
# 以用户ID为x轴，权重为y轴，绘制条形图
plt.figure(figsize=(10, 6))
influential_users = df.groupby('用户名')['权重'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=influential_users.index, y=influential_users.values, palette=palette)
plt.title('Top 10 Influential Users')
plt.ylabel('Total Influence Weight')
plt.xticks(rotation=45)
plt.savefig(r"charts\trends\Top-10-Influential-Users.png")
plt.show()

# 4. 用户属性分析
# 性别分布
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='性别', palette=palette)
plt.title('Gender Distribution')
plt.savefig(r"charts\trends\Gender-Distribution.png")
plt.show()

# 账户类型分布
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='账户类型', palette=palette)
plt.title('Account Type Distribution')
plt.savefig(r"charts\trends\Account-Type-Distribution.png")
plt.show()


# 是否为机构号分布
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='是否机构号', palette=palette)
plt.title('Organizational Account Distribution')
plt.savefig(r"charts\trends\Organizational-Account-Distribution.png")
plt.close()


# 2. 关联规则分析（这里以性别和账户类型为例）
# 将数据转为适合关联规则分析的格式
df_association = pd.get_dummies(df[['性别', '账户类型']])
frequent_itemsets = apriori(df_association, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# 保存关联规则到CSV
rules.to_csv(r"charts\trends\association_rules.csv")

# 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='lift', palette=palette, sizes=(20, 200))
plt.title('Support vs Confidence')
plt.savefig(r"charts\trends\Support-vs-Confidence.png")
plt.show()

# 热图
pivot = rules.pivot(index='antecedents', columns='consequents', values='lift')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Lift'})
plt.title('Lift Heatmap')
plt.savefig(r"charts\trends\Lift-Heatmap.png")
plt.show()






# 创建一个ExcelWriter对象
with pd.ExcelWriter(r"charts\trends\chart_data.xlsx") as writer:
    # 保存趋势预测数据
    df.resample('M').size().to_excel(writer, sheet_name='Trend Over Time')
    
    # 保存用户活跃度数据
    active_users.to_excel(writer, sheet_name='Top 10 Active Users')
    
    # 保存用户影响力数据
    influential_users.to_excel(writer, sheet_name='Top 10 Influential Users')
    
    # 保存性别分布数据
    df['性别'].value_counts().to_excel(writer, sheet_name='Gender Distribution')
    
    # 保存账户类型分布数据
    df['账户类型'].value_counts().to_excel(writer, sheet_name='Account Type Distribution')
    
    # 保存是否为机构号分布数据
    df['是否机构号'].value_counts().to_excel(writer, sheet_name='Organizational Account Distribution')