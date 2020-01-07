import copy
import time

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from item_based_CF.item_rating_compare import ItemRatingCompare
from user_based_CF.compare_to import CompareTo
# data = pd.read_csv("resource/train.csv",parse_dates=['timestamp'])
# # 将样本分为x表示特征，y表示类别
# x, y = data.ix[:, 0:], data.ix[:, 0:]
# # 测试集为30%，训练集为70%
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
# user_names = x_train["userID"].unique()
# item_names = x_train["itemID"].unique()
# print(user_names)
# user_n = len(user_names)
# item_n = len(item_names)
# train = pd.DataFrame(np.zeros((user_n, item_n)), index=user_names, columns=item_names)
# train2 = pd.DataFrame(np.zeros((user_n, item_n)), index=user_names, columns=item_names)
#
# train = train.sort_index()
# train = train.sort_index(axis=1)
# for i in x_train.itertuples():
#      train.loc[getattr(i, "userID"), getattr(i, "itemID")] = getattr(i, "rating")
#
# # train.to_csv("resource/ItemTest.csv")
# start =time.clock()
# it=CompareTo(train,0,10,{}).get_front_n()
# end =time.clock()
# print("running time:%s"%(end-start))
from user_based_CF.user_based_factory import UserBasedFactory

data = pd.read_csv("resource/train.csv")
test = pd.read_csv("resource/test_index.csv")

train = np.zeros((2967, 4125))
for line in data.itertuples():
    train[line[1], line[2]] = line[3]

result = copy.deepcopy(train)
train[:, 0] = train[:, 0] + 0.00000001
average = np.average(train, axis=1, weights=np.int64(train > 0))
print("进入主函数")
# 30->3.6819
result = UserBasedFactory(train, result, average, 40).get_result()

records_predict = []
count = 0
for i in test.itertuples():
    try:
        ss = result[i[1], i[2]]
        records_predict.append([count, ss])
    except:
        records_predict.append([count, 0])
    count = count + 1

ss = pd.DataFrame(records_predict, columns=["dataID", "rating"])
ss.to_csv("resource/last_self_rating.csv", index=None)
