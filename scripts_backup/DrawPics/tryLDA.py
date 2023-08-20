import re
import jieba
import gensim
from gensim import corpora, models, similarities
import pandas as pd
import pyLDAvis.gensim_models


df0 = pd.read_csv('DataForPic.csv',encoding='ANSI')
df_comments = df0['回答内容']

'''df_weight = df0['影响力']*10
df_weight = df_weight.astype("int")
df_comments = df_comments * df_weight'''

lis_sentances = df_comments.to_list()
base_data = lis_sentances
#将base_data中的数据进行遍历后分词
base_items = [[i for i in jieba.lcut(item) if len(i)>1] for item in base_data]
print(base_items)

off = open('ChineseStopWords.txt', encoding='utf-8')
off = off.read()
temp_lis = []
for word_lis in base_items:
    temp_word_lis = []
    for word in word_lis:
        if word not in off and not bool(re.search('[a-z]', word)) and not bool(re.search(r'\d', word)):
            temp_word_lis.append(word)
    temp_lis.append(temp_word_lis)

base_items = temp_lis
print(base_items)

# 生成词典
dictionary = corpora.Dictionary(base_items)
dictionary.save("test.dict")
#可以保存词典，随后使用时再装载

# 通过doc2bow稀疏向量生成语料库
corpus = [dictionary.doc2bow(item) for item in base_items]
'''
print("第一个文档的向量为:{0}".format(corpus[0])) #查看语料库
print("第三个向量的词袋：", dictionary.doc2bow(base_items[2]))
'''

'''
#通过TF-IDF模型算法，计算出tf值
tf = models.TfidfModel(corpus)
tf.save("test.model")
#可以保存模型，随后使用时再装载
print("第3首诗的TF-IDF：\n", tf[corpus[2]])
'''

#训练LDA模型
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
num_topics=8) #所有主题的单词分布

#查看所有主题
for i,topic in enumerate(lda.print_topics(num_topics=8, num_words=5)):
    print("主题",i+1,": ",topic[1])

#查看某一文档的主题
print("文档的内容为：\n{0}".format(base_data[2]))
topic = lda.get_document_topics(corpus[2])
print("文档的主题分布为：\n{0}".format(topic))

for i in range(5):
    print("文档的内容为：\n{0}".format(base_data[i]))
    topic = lda.get_document_topics(corpus[i])
    print("文档的主题分布为：\n{0}".format(topic))
    print('#######')

pd.DataFrame(base_data).to_csv('topic.csv')

d=pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)

'''
lda: 计算好的话题模型

corpus: 文档词频矩阵

dictionary: 词语空间
'''
pyLDAvis.save_html(d, 'lda_pass101.html')
# pyLDAvis.show(d)		#展示在浏览器