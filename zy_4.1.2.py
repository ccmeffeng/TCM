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


def sim(vec, num):
    lst_arr = np.load('diseases.npy')
    inp_arr = np.array(vec)
    lst = []
    for i in range(len(lst_arr)):
        dis = np.linalg.norm(lst_arr[i] - inp_arr)
        lst.append((dis, i))
    lst.sort()
    sims = []
    for j in range(num):
        sims.append(lst[j])
    return sims


if __name__ == '__main__':
    sentences = open('diseases_out_description.txt', 'r', encoding='utf-8').readlines()
    inp = get_input('input.txt')
    mod = Doc2Vec.load('model_4.0.1.md')
    vec = sen2vec(mod, inp)
    siml = sim(vec, 5)
    for sim, index in siml:
        print('%d %f' %(index, sim))
        print(sentences[index])
