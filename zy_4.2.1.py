import re
import numpy as np
from gensim.models.doc2vec import Doc2Vec # 用于进行doc2vec学习，将文章段落转化为向量

def sen2vec(model, sentence):
    model.random.seed(0)
    vec = model.infer_vector(sentence)
    return vec


def ana2npy(file, model):
    yf_file = open(file, 'r', encoding='utf-8').readlines()
    matrix = []
    ids = 0
    for i in range(len(yf_file)):
        if re.search('\d+', yf_file[i]):
            ids = int(re.search('\d+', yf_file[i]).group())
        elif yf_file[i] != '\n' and yf_file[i-1] == '\n':
            vec = list(sen2vec(model, yf_file[i].strip('\n')))
            vec.append(ids)
            matrix.append(vec)
    return matrix

if __name__ == '__main__':
    mat = ana2npy('yaofang.txt', Doc2Vec.load('model_4.0.1.md'))
    print(mat[0])
    print('%d %d %f' %(len(mat),len(mat[0]),mat[0][0]))
    np.save('anagraph.npy', np.array(mat))
