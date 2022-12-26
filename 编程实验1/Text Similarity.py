import re
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora, models, similarities
import time


# 数据清洗
def textclear(filename):
    file = open(filename, 'r', encoding='gbk')  # 用gbk进行编码
    texts = file.readlines()
    # 写入新的文件
    filterd_file1 = open("1998new.txt", 'w', encoding='gbk')
    filterd_file2 = open("1998new.cut", 'w', encoding='gbk')
    for line in texts:
        if re.search(" ", line):
            line = re.sub("\d*-\d*-\d*-\d*/m ", "", line)  # 删去时间戳
            line = re.sub("[^a-z]*/[u,w,f,p,k,c,y]", "", line)  # 根据词性标注清洗
            line = re.sub("/[a-z,A-Z]*", "", line)  # 删去词性标注
            filterd_file1.write(line)
            filterd_file2.write(line)
        '''stop_words = set(stopwords.words('Chinese'))  # 停止词
        for w in line:  # 循环遍历，如果不属于停止词，则加入到过滤后的句子列表中
            if w not in stop_words:
                filterd_file1.write(w)'''
    return

#
class ToBows(object):
    def __init__(self, dict, in_file):
        self.dict = dict
        self.in_file = in_file

    def __iter__(self):
        for line in open(self.in_file, encoding='gbk'):
            yield self.dict.doc2bow(line.split())


if __name__ == '__main__':
    start = time.time()# 计时
    # 数据清洗
    textclear('199801_clear (1).txt')
    end = time.time()
    print("数据清洗用时为：", end - start, "s")
    is_train = False
    # 进行训练计算模型
    start = time.time()
    if is_train:
        # 建立词典
        dict=corpora.Dictionary(line.lower().split() for line in open('1998new.txt', encoding='gbk'))
        dict.save('1998new.dic')
        print("***********基本信息***********")
        print('词数：', len(dict.keys()))
        print('处理的文档数:', dict.num_docs)
        bows = ToBows(dict, in_file='1998new.cut')# 建立词袋语料
        corpora.MmCorpus.serialize('1998new.mm', bows)# 保存词袋信息
        # 计算iftdf
        tfidf = models.TfidfModel(dictionary=dict)
        corpus_tfidf = tfidf[bows]
        tfidf.save('1998new.tfidf')
        # 计算lsi模型并保存
        lsi = models.LsiModel(corpus_tfidf, id2word=dict, num_topics=200)#选择维度为200
        lsi.save('1998new.lsi')
        corpus_lsi = lsi[corpus_tfidf] # 计算所有语料
        # 生成相似矩阵
        bows = corpora.MmCorpus('1998new.mm')  # 加载bows
        tfidf = models.TfidfModel.load('1998new.tfidf')  # 加载tfidf模型
        lsi = models.LsiModel.load('1998new.lsi')  #加载LSI模型
        Similarmatrix = similarities.MatrixSimilarity(lsi[tfidf[bows]])
        Similarmatrix.save('1998new.Similarmatrix')  # 保存相似矩阵
        print("***********相似矩阵为***********\n",Similarmatrix)
        end = time.time()
        print("相似度模型训练时间为：", end - start, "s")
    # 应用模型,相关的查询
    else:
        dict = corpora.Dictionary.load('1998new.dic')#加载词典
        tfidf = models.TfidfModel.load('1998new.tfidf')  #加载tfidf模型
        lsi = models.LsiModel.load('1998new.lsi')  #加载LSI模型
        Similarmatrix = similarities.MatrixSimilarity.load('1998new.Similarmatrix')  #加载相似矩阵
        # 测试文档
        test_doc = """台湾  是  中国  领土  不可分割  一  部分  完成  祖国  统一  是  大势所趋  民心所向  任何  企图  制造  两  个  中国  一中一台  台湾  独立  图谋  都  注定  要  失败  希望  台湾  当局  民族  大义  为重  拿  出  诚意  采取  实际  行动  推动  两岸  经济  文化  交流  人员  往来  促进  两岸  直接  通邮  通航  通商  早日  实现  尽早  回应  我们  发出  一个  中国  原则  两岸  进行  谈判  郑重  呼吁"""
        test_bow = dict.doc2bow(test_doc.split()) # 把测试语料转成词袋向量
        test_tfidf = tfidf[test_bow]# 求tfidf值
        test_lsi = lsi[test_tfidf]# 转成lsi向量
        # 求解相似性文档
        similarity_res = Similarmatrix[test_lsi]
        print('*********结果为***********\n',sorted(enumerate(similarity_res), key=lambda item: -item[1]))#进行排序
        end = time.time()
        print("相似度计算时间为：", end - start, "s")
