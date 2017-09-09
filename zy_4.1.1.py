import re  # 匹配正则表达式，用于查找
import numpy as np
from gensim.models.doc2vec import Doc2Vec # 用于进行doc2vec学习，将文章段落转化为向量


# 从文件中按行读取数据并且去除空行，存储为一个已分词的list，返回该list
def get_data(in_file):
    data = []
    stopwords = open('stopword.txt', 'r', encoding='utf-8').read()
    stopwords = re.split(r'\n', stopwords)
    for i in open(in_file, 'r', encoding='utf-8').readlines():
        if i != '\n':
            i = re.sub(r'\n', '', i)
            cor = []
            for s in i:
                if s not in stopwords:
                    cor.append(s)
            data.append(cor)
    return data


def sen2vec(model, sentence):
    model.random.seed(0)
    vec = model.infer_vector(sentence)
    return vec


if __name__ == '__main__':
    d = get_data('diseases_out_description.txt')
    lst = []
    mod = Doc2Vec.load('model_4.0.1.md')
    for line in d:
        lst.append(sen2vec(mod,line))
    arr = np.array(lst)
    np.save('diseases.npy', arr)

