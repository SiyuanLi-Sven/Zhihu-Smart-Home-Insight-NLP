import wordcloud
import pandas as pd
import jieba

# 用户id	用户名	用户头像	发布时间	性别	账户类型	是否机构号	粉丝数量	点赞数量	评论数	回答内容
df0 = pd.read_csv('DataForPic.csv',encoding='ANSI')
df_comments = df0['回答内容']
df_weight = df0['影响力']*10
df_weight = df_weight.astype("int")
df_comments = df_comments * df_weight
f = ''.join(df_comments.to_list())

# print(f)
f = jieba.lcut(f)
ft = []
for i in f:
    if len(i) > 1:
        ft.append(i)
f = ft
# print(f)
f = ' '.join(f)

stopwords = set()
content = [line.strip() for line in open('ChineseStopWords.txt','r',encoding='utf-8').readlines()]
stopwords.update(content)
print(stopwords)

w = wordcloud.WordCloud(font_path='C:\Windows\Fonts\SIMHEI.TTF',
                        width=1600,
                        height=900,
                        max_words=20,
                        stopwords=stopwords,
                        background_color='White',
                        collocations=False
                        )
w.generate(f)
w.to_file('savedWordcloud/wordcloud001.jpg')