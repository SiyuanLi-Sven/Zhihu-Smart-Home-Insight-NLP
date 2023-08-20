import pandas as pd
import os
import numpy as np
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from snownlp import SnowNLP
import seaborn as sns

class DataAnalysis:
    def __init__(self, name):
        self.name = name
        self.df = self.load_data()
        self.df_weighted = self.assign_weights()

    def load_data(self):
        path = 'spider_output/{}'.format(self.name)
        all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.xlsx')]
        
        # 读取并合并所有Excel文件
        df_list = [pd.read_excel(file) for file in all_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # 输出总行数
        total_rows = df.shape[0]
        print(f"Total number of rows: {total_rows}")
        
        # 输出“回答内容”列整体总字符数
        total_chars = df['回答内容'].str.len().sum()
        print(f"Total number of characters in '回答内容' column: {total_chars}")
        
        return df

    def assign_weights(self):
        # 使用ln(x-1)计算权重，确保x>1，否则设置为0
        self.df['粉丝权重'] = self.df['粉丝数量'].apply(lambda x: np.log(x-1) if x > 1 else 0)
        self.df['权重'] = np.sqrt(self.df['粉丝权重']) + np.sqrt(self.df['点赞数量']) +  np.sqrt(self.df['评论数'])*10
        # 限制权重的最大值为10
        self.df['权重'] = self.df['权重'].clip(upper=15)
        print(self.df['权重'].describe())
        
        return self.df


    def generate_wordcloud(df,name):
        # 去除"回答内容"列中的空值
        df = df.dropna(subset=['回答内容'])

        # 使用权重对回答内容进行加权
        weighted_texts = ''
        for index, row in df.iterrows():
            weighted_texts += (row['回答内容'] + ' ') * int(row['权重'])

        # 读取附加的词库并添加到jieba分词器中
        with open('resource/JiebaAddwords.txt', 'r', encoding='utf-8') as f:
            for word in f.readlines():
                jieba.add_word(word.strip())

        # 读取停用词
        with open('resource/JiebaStopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())

        # 使用jieba进行分词
        seg_list = jieba.lcut(weighted_texts, cut_all=False)
        seg_list = [word for word in seg_list if word not in stopwords and len(word) > 1]

        # 生成词云
        wc = WordCloud(
            font_path='simhei.ttf', 
            background_color='white', 
            width=800, 
            height=600, 
            max_font_size=100, 
            max_words=50, 
            collocations=False,
            stopwords=stopwords
                    )
        wc.generate(' '.join(seg_list))

        # 显示词云
        plt.figure(figsize=(10, 8), dpi=300)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('charts/wordclouds/{}-wordclouds.png'.format(name))
        plt.show()

    def sentiment_analysis(df,name):
        sentiments = []
        total_rows = len(df)
        
        for index, row in df.iterrows():
            if pd.notnull(row['回答内容']):
                sentiment = SnowNLP(row['回答内容']).sentiments
            else:
                sentiment = None
            sentiments.append(sentiment)
            
            # 打印进度
            if (index + 1) % 100 == 0:  # 每处理100条数据打印一次进度
                print(f"Processed {index + 1} out of {total_rows}. Remaining: {total_rows - (index + 1)}")
        
        df['sentiment'] = sentiments

        # 使用seaborn绘制情感分析值的分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(df['sentiment'].dropna(), kde=True, bins=30, label='Sentiment Distribution')
        plt.axvline(df['sentiment'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Sentiment')
        plt.title('Sentiment Analysis Distribution')
        plt.xlabel('Sentiment Value')
        plt.ylabel('Density')
        plt.xlim([df['sentiment'].min() - 0.05, df['sentiment'].max() + 0.05])  # 动态调整x轴范围
        plt.legend()
        plt.savefig('charts/sentiments/{}-sentiments.png'.format(name))
        plt.show()


# 示例使用
if __name__ == "__main__":
    name_lis = ['smart_home_ques','smart_funi_ques']
    for name in name_lis:
        analysis = DataAnalysis(name)
        analysis.generate_wordcloud()
        analysis.sentiment_analysis()
