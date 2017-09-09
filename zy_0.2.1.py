import re
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


model = Doc2Vec.load('model_0.0.1.md')
d = get_data('medicine.txt')
out_a = open('X.txt', 'w', encoding='utf-8')

s1, s2, mdc = [], [], []
for line in d:
    if not re.search('【', line):
        s1.append(line)
    elif re.search('【组成】', line):
        line = re.sub('【组成】', '', line)
        m = line.split()
        s2.append(m[0])
        for x in m:
            if x not in mdc:
                mdc.append(x)
    elif re.search('【主治】', line):
        line = re.sub('【主治】', '', line)
        out_a.write(str(model.infer_vector(wash_data(line),alpha=0,min_alpha=0,steps=1))+'\n')
out_a.close()

out_b = open('Y.txt', 'w', encoding='utf-8')
out_b.write(str(mdc)+'\n')

for i in range(len(s2)):
    y = [0] * len(mdc)
    j = mdc.index(s2[i])
    y[j] = 1
    out_b.write(s1[i]+'\n')
    out_b.write(str(y)+'\n')
out_b.close()
