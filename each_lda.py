import func_wash_wordcloud_pylda as tools
import pandas as pd
import re
import jieba
import gensim
from gensim import corpora, models, similarities
import pandas as pd
import pyLDAvis.gensim_models


def preprocess_data(df: pd.DataFrame) -> list:
    """预处理数据，进行分词"""
    # 读取附加的词库并添加到jieba分词器中
    with open('resource/JiebaAddwords.txt', 'r', encoding='utf-8') as f:
        for word in f.readlines():
            jieba.add_word(word.strip())
    lis_sentences = df['回答内容'].astype(str).tolist()  # 将回答内容转换为字符串
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


# 示例使用
# df = load_data('DataForPic.csv')

name='smart_home_and_funi_ques'
df0 = tools.assign_weights(tools.load_data(name),name)
df = df0

base_items = preprocess_data(df)
base_items = remove_stopwords(base_items, 'resource\JiebaStopwords.txt')
dictionary, corpus = build_corpus(base_items)
lda = train_lda(corpus, dictionary)
display_topics(lda)
save_ldavis(lda, corpus, dictionary, r'charts\pylda\lda_{}.html'.format(name))