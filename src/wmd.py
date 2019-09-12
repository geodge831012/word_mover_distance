# -*- coding: utf-8 -*-

from gensim.models import Word2Vec

import numpy as np
import jieba_fast
import pickle
from jieba_fast import analyse
#import jieba_fast.posseg as pseg

jieba_fast.analyse.set_stop_words('../data/stopwords.txt')
jieba_fast.add_word('溢缴款')

# word2vec模型
model = Word2Vec.load("../word2vec/word2vec_wx")



#########################################################################
## main 主函数 ##

if __name__ == '__main__':

    l  = word_list = list(jieba_fast.cut("溢缴单笔领回限额"))
    l0 = word_list = list(jieba_fast.cut("溢缴单笔领回限额多少"))
    l1 = word_list = list(jieba_fast.cut("溢缴单笔领回限额是多少"))
    l2 = word_list = list(jieba_fast.cut("溢缴款领回手续费如何收费"))
    l3 = word_list = list(jieba_fast.cut("白金尊贵卡年费如何用积分兑换"))

    print(type(l))
    print(l)
    print(type(l0))
    print(l0)
    print(type(l1))
    print(l1)
    print(type(l2))
    print(l2)
    print(type(l3))
    print(l3)

    distance = model.wmdistance(l1, l)
    print(distance)
    distance = model.wmdistance(l, l1)
    print(distance)

    print("=====================================================================")

    distance = model.wmdistance(l1, l0)
    print(distance)
    distance = model.wmdistance(l0, l1)
    print(distance)

    print("=====================================================================")

    distance = model.wmdistance(l1, l2)
    print(distance)
    distance = model.wmdistance(l2, l1)
    print(distance)

    print("=====================================================================")

    distance = model.wmdistance(l1, l1)
    print(distance)

    print("=====================================================================")

    distance = model.wmdistance(l1, l3)
    print(distance)
    distance = model.wmdistance(l3, l1)
    print(distance)

    #print(get_wmd("溢缴单笔领回限额是多少", "溢缴款领回手续费如何收费"))
    #print(get_wmd("溢缴单笔领回限额是多少", "白金尊贵卡年费如何用积分兑换"))
