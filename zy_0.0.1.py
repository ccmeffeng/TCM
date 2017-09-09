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
    model.save('model_1.0.1.md')
    return model


# 寻找最接近的10个
def sim(model, sentence):
    raw = wash_data(sentence)
    model.random.seed(0)
    vec = model.infer_vector(raw)
    sims = model.docvecs.most_similar([vec], topn=10)
    return sims


d = get_data('diseases_out_description.txt') + get_data('0001.txt') + get_data('0002.txt') + get_data('0005.txt')
mod = train(d, 300, 5, 3)

for i in range(10):
    test_data = d[i]
    print(sim(mod, test_data))