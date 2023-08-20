# -*- coding: utf-8 -*-
import os
import time
import requests
import csv

zhihuColumn = "c_1034016963944755200" # 自行替换专栏编号,此处是自娱自乐的游戏访谈录(在URL中的编号)
startURL = "https://www.zhihu.com/api/v4/columns/" + zhihuColumn + "/items?limit=10&offset={}"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0"}

# 输入URL,得到JSON数据
def getJSON(url):
    try:
        r = requests.get(url, headers = headers)
        r.raise_for_status() # 响应状态码,出错则抛出异常
        r.encoding = r.apparent_encoding
        return r.json()
    except Exception as ex:
        print(type(ex))
        time.sleep(10)
        return getJSON(url)

# 输入文章总数,输出所有文章标题和链接的CSV文件
def process(total):
    num = 0 # 文章编号
    if (os.path.exists("zhihu.csv")): # 已经存在时则删除
        os.remove("zhihu.csv") 
    with open("zhihu.csv", "a", encoding = "UTF-8", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["编号", "标题", "链接"]) # csv表头部分
        for offset in range(0, total, 10):
            jsonData = getJSON(startURL.format(offset))
            items = jsonData["data"]
            for item in items:
                num = num + 1
                writer.writerow([num, item["title"], item["url"]])

if __name__ == "__main__":
    jsonData = getJSON(startURL.format(0))
    process(jsonData["paging"]["totals"])
