import re  # 匹配正则表达式，用于查找
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
    stopwords = re.split(r'\n', stopwords)
    corpus = []
    for i in seg_list:
        if i not in stopwords:
            corpus.append(i)
    return corpus


# 数据打上tag
def tag(raws):
    corpora_documents = []
    for i, item_text in enumerate(raws):
        words_list = wash_data(item_text)
        document = TaggedDocument(words=words_list, tags=[i])
        corpora_documents.append(document)
    return corpora_documents


# doc2vec可以用神经网络算法将文本（有tag标注的、经过分词之后的句子的列表）转化为向量，这个函数返回一个model
def train(raw_documents, siz, wdw, epoch):
    # 建立doc2vec模型
    # 可调节参数：siz表示向量维数，window表示窗口大小（训练一个词的时候考虑最靠近的多少个词）
    model = Doc2Vec(size=siz, window=wdw, min_count=1, dm=1, iter=10)
    # 使用所有的数据建立词典
    s_data = tag(raw_documents)
    model.build_vocab(s_data)
    # 进行训练
    for i in range(epoch):
        model.train(s_data)
    # 保存模型
    model.save('model_0.0.2.md')
    return model


# 寻找最接近的10个
def sim(model, index):
    sentence = d0[index]
    raw = wash_data(sentence)
    model.random.seed(0)
    vec = model.infer_vector(raw)
    sims = model.docvecs.most_similar([vec], topn=20)
    jmax, jmin = 0, 0
    for i in range(20):
        if sims[i][0] == index:
            jmax = sims[i][1]
        elif sims[i][0] < 990 and jmin != 0:
            jmin = sims[i][1]
        if jmax*jmin:
            break
        elif i == 19:
            if not jmax:
                jmax = sims[i][1]
            if not jmin:
                jmin = sims[i][1]
    return jmax-jmin


d0 = get_data('diseases_out_description.txt')
for i in range(32):
    d1, d2, d3, d4, d5 = [], [], [], [], []
    a, b, c, d, e = 0, 0, 0, 0, 0
    if i % 2 == 1:
        e = 1
        d5 = get_data('0005.txt')
    if i % 4 >= 2:
        d = 1
        d4 = get_data('0004.txt')
    if i % 8 >= 4:
        c = 1
        d3 = get_data('0003.txt')
    if i % 16 >= 8:
        b = 1
        d2 = get_data('0002.txt')
    if i >= 16:
        a = 1
        d1 = get_data('0001.txt')
    print([a, b, c, d, e], end='  distance = ')
    d = d0 + d1 + d2 + d3 + d4 + d5
    mod = train(d, 300, 5, 3)

    distance = 0
    for j in range(990):
        distance += sim(mod, j)
    print(distance)
