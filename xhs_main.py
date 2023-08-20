import pandas as pd
import numpy as np
import jieba
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import gensim
from gensim import corpora, models, similarities
import func_wash_wordcloud_pylda as tools
import re


def read_data():
    df = pd.read_excel('resource\全屋智能家居from小红书.xlsx')
    # 去除任何一项为空的行
    df.dropna(inplace=True)
    return df

# 对喜欢列进行预处理
def preprocess_likes(value):
    if '万' in str(value):
        return float(value.replace('万', '')) * 10000
    else:
        return float(value)

# 第二个函数：对回答进行加权
def assign_weights(df):
    # 预处理喜欢列
    df['喜欢'] = df['喜欢'].apply(preprocess_likes)
    
    # 计算权重
    df['权重'] = df['喜欢'].apply(lambda x: np.log(x-1) if x > 1 else 0)
    return df

# 第三个函数：绘制词云图
def generate_wordcloud(df):
    # 使用权重进行加权合并
    weighted_text = ''
    for index, row in df.iterrows():
        weighted_text += (row['文本'] + ' ') * int(row['权重'])
    
    # 使用wordcloud绘制词云
        # 读取附加的词库并添加到jieba分词器中
    with open('resource/JiebaAddwords.txt', 'r', encoding='utf-8') as f:
        for word in f.readlines():
            jieba.add_word(word.strip())

    # 读取停用词
    with open('resource/JiebaStopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())

    # 使用jieba进行分词
    seg_list = jieba.lcut(weighted_text, cut_all=False)
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
    plt.savefig(r'xhs_output\xhs_wordcloud.jpg')
    plt.show()


def preprocess_data(df: pd.DataFrame) -> list:
    """预处理数据，进行分词"""
    # 读取附加的词库并添加到jieba分词器中
    with open('resource/JiebaAddwords.txt', 'r', encoding='utf-8') as f:
        for word in f.readlines():
            jieba.add_word(word.strip())
    lis_sentences = df['文本'].astype(str).tolist()  # 将帖子内容转换为字符串
    base_items = [[i for i in jieba.lcut(item) if len(i)>1] for item in lis_sentences]
    return base_items


def remove_stopwords(base_items: list, stopwords_path: str) -> list:
    """移除停用词"""
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    filtered_items = []
    for word_list in base_items:
        filtered_list = [word for word in word_list if word not in stopwords and not bool(re.search('[a-z]', word)) and not bool(re.search(r'\d', word))]
        filtered_items.append(filtered_list)
    return filtered_items

def build_corpus(base_items: list) -> (gensim.corpora.Dictionary, list):
    """构建语料库"""
    dictionary = corpora.Dictionary(base_items)
    corpus = [dictionary.doc2bow(item) for item in base_items]
    return dictionary, corpus

def train_lda(corpus: list, dictionary: gensim.corpora.Dictionary, num_topics: int = 8) -> gensim.models.ldamodel.LdaModel:
    """训练LDA模型"""
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    return lda

def display_topics(lda: gensim.models.ldamodel.LdaModel, num_topics: int = 8, num_words: int = 5):
    """显示所有主题"""
    for i, topic in enumerate(lda.print_topics(num_topics=num_topics, num_words=num_words)):
        print("主题", i+1, ": ", topic[1])

def save_ldavis(lda: gensim.models.ldamodel.LdaModel, corpus: list, dictionary: gensim.corpora.Dictionary, output_path: str):
    """保存LDA可视化结果为HTML"""
    vis_data = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)
    pyLDAvis.save_html(vis_data, output_path)



# 主程序
if __name__ == "__main__":
    # 读取数据
    df = read_data()
    
    # 赋予权重
    df_weighted = assign_weights(df)
    
    # 绘制词云图
    generate_wordcloud(df_weighted)

    # 示例使用
    df =  df_weighted
    base_items = preprocess_data(df)
    base_items = remove_stopwords(base_items, 'resource\JiebaStopwords.txt')
    dictionary, corpus = build_corpus(base_items)
    lda = train_lda(corpus, dictionary)
    display_topics(lda)
    save_ldavis(lda, corpus, dictionary, r'xhs_output\lda_pass101.html')
