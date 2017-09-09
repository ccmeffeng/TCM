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


def sim(model, index):
    sentence = d0[index]
    raw = wash_data(sentence)
    model.random.seed(0)
    vec = model.infer_vector(raw)
    sims = model.docvecs.most_similar([vec], topn=5)
    print(sims)
    jmax, jmin = 0, 0
    for i, j in sims:
        if i == index:
            jmax = j
        elif i < 990 and jmin == 0:
            jmin = j
        if jmax != 0 and jmin != 0:
            break
    if not jmax:
        jmax = 0
    if not jmin:
        jmin = 0
    return jmax, jmin

d0 = get_data('diseases_out_description.txt')
model = Doc2Vec.load('model_0.0.1.md')

for i in range(10):
    print(sim(model, i))

