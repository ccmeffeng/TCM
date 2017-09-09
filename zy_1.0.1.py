import re
import jieba  # 结巴分词，将中文句子分成词语列表
from gensim.models.doc2vec import Doc2Vec  # 用于进行doc2vec学习，将文章段落转化为向量


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


def vec_cal(s1):
    ss1 = wash_data(s1)
    model.random.seed(0)
    v1 = model.infer_vector(ss1)
    return str(v1)

model = Doc2Vec.load('model_1.0.1.md')

d = get_data('diseases_out_description.txt')
ot = open('vector.txt', 'w', encoding='utf-8')
for s in d:
    ot.write(vec_cal(s)+'\n')
ot.close()
