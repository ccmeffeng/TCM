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


def most_sim(s, k):
    def get_vecs():
        docvec, vct = [], []
        for line in open('vector.txt', 'r', encoding='utf-8').readlines():
            line = line.replace('[', '')
            line = line.replace(']', '')
            if len(vct) == 300:
                docvec.append(vct)
                vct = []
            for dta in line.split():
                if dta != 'end':
                    vct.append(float(dta))
        return docvec

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

    ss = wash_data(s)
    model.random.seed(0)
    vv = model.infer_vector(ss)
    sim, tt = [], get_vecs()
    for i in range(len(tt)):
        sim.append((mul(vv, tt[i]), i))
    sim.sort(reverse=True)
    sims = []
    for j in range(k):
        s = '%.3f' % sim[j][0]
        sims.append((sim[j][1], s))
    return sims


def sim_cal(s1, s2):
    ss1, ss2 = wash_data(s1), wash_data(s2)
    model.random.seed(0)
    v1 = model.infer_vector(ss1)
    model.random.seed(0)
    v2 = model.infer_vector(ss2)
    x1, x2, x = 0, 0, 0
    for i in range(len(v1)):
        x1 += v1[i] * v1[i]
        x2 += v2[i] * v2[i]
        x += v1[i] * v2[i]
    siml = x / math.sqrt(x1 * x2)
    return siml

model = Doc2Vec.load('model_1.0.1.md')

#d = get_data('diseases_out_description.txt')
#d1 = d[0]
#d2 = open('diseases_out_description.txt','r',encoding='utf-8').readlines()[100]

d1 = '我感觉头很疼，比较困'
d2 = '我感觉比较困，头很疼'
#print(most_sim(d1, 10))
#print(most_sim(d2, 10))

print(wash_data(d1))
print(wash_data(d2))

print(sim_cal(d1,d2))
