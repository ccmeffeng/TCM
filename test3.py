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


def sim(model, sentence):
    raw = wash_data(sentence)
    vec = model.infer_vector(raw)
    sims = model.docvecs.most_similar([vec], topn=5)
    return sims


model = Doc2Vec.load('model_0.0.1.md')

s = '使用了一些数据进行测试'
s1 = '使用了一些'
s2 = '数据进行测试'

print(sim(model,s))
