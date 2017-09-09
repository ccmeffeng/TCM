import math
import jieba  # 结巴分词，将中文句子分成词语列表
from gensim.models.doc2vec import Doc2Vec  # 用于进行doc2vec学习，将文章段落转化为向量


def mul(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('Not Equal Length')
    x, x1, x2 = 0, 0, 0
    for i in range(len(v1)):
        x += v1[i] * v2[i]
        x1 += v1[i] * v1[i]
        x2 += v2[i] * v2[i]
    m = x / math.sqrt(x1 * x2)
    return m


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
    return v1


def get_vecs():
    docvec, vct = [], []
    for line in open('vector.txt','r',encoding='utf-8').readlines():
        line = line.replace('[','')
        line = line.replace(']', '')
        if len(vct) == 300:
            docvec.append(vct)
            vct = []
        for dta in line.split():
            if dta != 'end':
                vct.append(float(dta))
    return docvec


def most_sim(s, k):
    ss = wash_data(s)
    model.random.seed(0)
    vv = model.infer_vector(ss)
    sim, tt = [], get_vecs()
    for i in range(len(tt)):
        sim.append((mul(vv, tt[i]), i))
    sim.sort(reverse=True)
    sims = []
    for j in range(k):
        s = '%.3f'%sim[j][0]
        sims.append((sim[j][1], s))
    return sims


model = Doc2Vec.load('model_1.0.1.md')
#s0 = open('diseases_out_description.txt','r',encoding='utf-8').readlines()[100]
s0 = str(input('输入描述:\n'))
print(most_sim(s0, 10))
