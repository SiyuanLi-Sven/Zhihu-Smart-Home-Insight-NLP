#coding:UTF-8
from snownlp import SnowNLP
import pandas as pd

df0 = pd.read_csv('comments.csv')
for index,row in df0.iterrows():
    s = SnowNLP(df0['内容'][index])
    print(s.sentiments)
    df0.loc[index,'sentiments'] = s.sentiments

df0.to_csv('snow_output.csv',encoding='utf-8')
