import re  # 匹配正则表达式，用于查找
import numpy as np
from gensim.models.doc2vec import Doc2Vec # 用于进行doc2vec学习，将文章段落转化为向量


# 从文件中按行读取数据并且去除空行，存储为一个已分词的list，返回该list
def get_input(in_file):
    data = []
    stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
    stopwords = re.split(r'\n', stopwords)
    for i in open(in_file, 'r', encoding='utf-8').read():
        if i != '\n' and i not in stopwords:
            data.append(i)
    return data


def sen2vec(model, sentence):
    model.random.seed(0)
    vec = model.infer_vector(sentence)
    return vec


def sim(vec, num, id):
    tot_lst = np.load('anagraph.npy')
    lst = []
    for arr in tot_lst:
        if arr[-1] == id:
            lst.append(list(arr)[0:-1])
    inp_arr = np.array(vec)
    lst_arr = np.array(lst)
    lst_sim = []
    for i in range(len(lst_arr)):
        dis = np.linalg.norm(lst_arr[i] - inp_arr)
        lst_sim.append((dis, i))
    lst_sim.sort()
    sims = []
    for j in range(num):
        if j < len(lst_sim):
            sims.append(lst_sim[j])
    return sims


def ana2print(ids, index):
    ana = open('yaofang.txt', 'r', encoding='utf-8').readlines()
    srch, count = 0, 0
    for i in range(len(ana)):
        if re.search('\d+', ana[i]):
            srch += 1
            count = 0
        elif ids == srch:
            if i > 0 and ana[i] != '\n' and ana[i-1] == '\n':
                count += 1
            if index == count:
                return ana[i] + ana[i+1] + ana[i+2] + ana[i+3]
        elif ids < srch:
            return None
    return None


if __name__ == '__main__':
    inp = get_input('input.txt')
    mod = Doc2Vec.load('model_4.0.1.md')
    vec = sen2vec(mod, inp)
    ids = 1
    siml = sim(vec, 5, ids)
    print(siml)
    for sim, index in siml:
        print(sim)
        print(ana2print(ids, index+1))
