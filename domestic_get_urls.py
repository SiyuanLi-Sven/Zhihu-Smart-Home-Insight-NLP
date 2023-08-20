from bs4 import BeautifulSoup

# 读取HTML文件
with open('pages\智能家居 - 搜索结果 - 知乎.html', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用BeautifulSoup解析HTML内容
soup = BeautifulSoup(content, 'html.parser')

# 提取所有的<a>标签
links = soup.find_all('a')

# 定义我们要查找的字段顺序
fields = ['/p/', 'zhuanlan', 'answer']

# 创建一个字典来存储链接，以字段为键，链接列表为值
links_by_field = {field: [] for field in fields}

# 将链接按字段分类
for link in links:
    href = link.get('href')
    if href:
        for field in fields:
            if field in href:
                links_by_field[field].append(href)
                break  # 只添加到一个字段中，避免重复

# 收集问题ID和zhuanlan链接
question_ids = []
zhuanlan_links = []

# 提取问题ID和记录zhuanlan链接
for field in fields:
    for link in links_by_field[field]:
        if "answer" in link:
            start_index = link.find('/question/') + len('/question/')
            end_index = link.find('/', start_index)
            question_id = link[start_index:end_index]
            question_ids.append(question_id)
        elif "zhuanlan" in link:
            zhuanlan_links.append(link)

# 打印问题ID
print("Question IDs:")
for question_id in question_ids:
    print(question_id)

# 打印zhuanlan链接
print("\nZhuanlan Links:")
for link in zhuanlan_links:
    print(link)
