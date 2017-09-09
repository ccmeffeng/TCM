import re
import math
import jieba  # 结巴分词，将中文句子分成词语列表
from gensim.models.doc2vec import Doc2Vec, TaggedDocument  # 用于进行doc2vec学习，将文章段落转化为向量


# 从文件中按行读取数据并且去除空行，存储为一个list，返回该list
def get_data(in_file):
    data = []
    for i in open(in_file, 'r', encoding='utf-8').readlines():
        if i != '\n':
            i = re.sub(r'\n', '', i)
            data.append(i)
    return data


def wash_data(sentence):
    # 结巴分词
    seg_list = jieba.cut(sentence)
    # 去除停用词
    stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
    stopwords = stopwords.split('\n')
    corpus = []
    for i in seg_list:
        if i not in stopwords:
            corpus.append(i)
    return corpus


def sim_cal(s1, s2):
    ss1, ss2 = wash_data(s1), wash_data(s2)
    v1, v2 = model.infer_vector(ss1, alpha=0, min_alpha=0,steps=1), model.infer_vector(ss2, alpha=0, min_alpha=0,steps=1)
    x1, x2, x = 0, 0, 0
    for i in range(len(v1)):
        x1 += v1[i] * v1[i]
        x2 += v2[i] * v2[i]
        x += v1[i] * v2[i]
    siml = x / math.sqrt(x1 * x2)
    return siml


model = Doc2Vec.load('model_0.0.1.md')
d = get_data('medicine.txt')

s1, s2, s3 = [], [], []
for line in d:
    if not re.search('【', line):
        s1.append('【'+line+'】')
    elif re.search('【组成】', line):
        line = re.sub('【组成】', '', line)
        s2.append(line)
    elif re.search('【主治】', line):
        line = re.sub('【主治】', '', line)
        s3.append(line)

s = '外感风寒表实证。证见恶寒发热，无汗咳喘，苔薄白，脉浮紧。'

res = []
for i in range(len(s3)):
    res.append((sim_cal(s, s3[i]), i))
res.sort(reverse = True)

for j in range(5):
    ind = res[j][1]
    print('%d %f' % (ind, res[j][0]))
    print(s1[ind] + ' ' + s2[ind])

print(sim_cal(s, s))
