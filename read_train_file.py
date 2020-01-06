import copy
import time

import math
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np

from item_based_CF.item_based_factory import ItemBasedFactory
from more.PMF_ALS import PMF
from user_based_CF.compare_to import CompareTo
from user_based_CF.user_based_factory import UserBasedFactory


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def get_average(records):
    """
    平均值
    """
    return sum(records) / len(records)


def get_train_matrix_and_test(train_matrix, test_, n_splits):
    for count in range(1, n_splits + 1):

        train = np.loadtxt("resource/train_matrix" + str(count) + ".csv", delimiter=",")
        test = pd.read_csv("resource/test" + str(count) + ".csv")
        #print(train.columns)
        # print(train.columns)
        # print(train.head())
        # print(train.loc[1])
        # print(train.loc[:, 1])
        train_matrix.append(train)
        test_.append(test)
    print("交叉测试集导入完成")


def creat_train_matrix_and_test(train_matrix, test_, n_splits):
    data = pd.read_csv("resource/train.csv")
    kf = KFold(n_splits=n_splits, shuffle=True)
    count = 0
    print("交叉测试集生成")
    for train_index, test_index in kf.split(data):
        count = count + 1
        print("第%s交叉测试集生成：" % count)
        train_data = data.loc[train_index]
        test = data.loc[test_index]
        #test.to_csv("resource/test" + str(count) + ".csv")
        #train_data.to_csv("resource/train" + str(count) + ".csv")
        train = np.zeros((2967,4125))

        print("原始数据生成矩阵中......")
        for line in train_data.itertuples():
            train[line[1], line[2]] = line[3]
        print("原始数据生成矩阵over")

        np.savetxt('resource/train_matrix' + str(count) + ".csv", train, delimiter=',')
        #train.to_csv("resource/rating" + str(count) + ".csv")
        print("矩阵集导出")
        test.to_csv("resource/test" + str(count) + ".csv")
        print("测试集导出")
        train_matrix.append(train)
        test_.append(test)



def user_rmse(count, train_matrix_now, test_now, final_user_RMSE):
    print("****************")
    print("USER_BASED开始")
    result = copy.deepcopy(train_matrix_now)
    train_matrix_now[:,0]=train_matrix_now[:,0]+0.00000001
    average=np.average(train_matrix_now, axis=1, weights=np.int64(train_matrix_now > 0))
    print("进入主函数")
    #30->3.6819
    result = UserBasedFactory(train_matrix_now, result, average,40).get_result()
    print("出去主函数")
    try:
     np.savetxt("resource/result_user" + str(count) + ".csv", result, delimiter=',')
    except Exception as e:
        print(e)
    print("结果矩阵集导出")
    records_real = []
    records_predict = []
    for i in test_now.itertuples():
        try:
            ss = result[i[1], i[2]]
            records_predict.append(ss)
            records_real.append(getattr(i, "rating"))
        except:
            records_predict.append(0)
            records_real.append(getattr(i, "rating"))
    final_user_RMSE.append(get_rmse(records_real, records_predict))
    print("第%s次交叉测试user_rmse为%s" % (count, final_user_RMSE[count - 1]))
    print("USER_BASED结束")
    print("****************")

def PMF_rmse(train_matrix_now):
    user = np.full((5, 2967), 0.5)
    item = np.full((5, 4125), 0.5)
    get_matrix_rating = PMF(item,user,train_matrix_now)

def item_rmse(count, train_matrix_now, test_now, final_item_RMSE):
    print("****************")
    print("ITEM_BASED开始")
    result = copy.deepcopy(train_matrix_now)
    train_matrix_now[0] = train_matrix_now[0] + 0.00000001
    average = np.average(train_matrix_now, axis=0, weights=np.int64(train_matrix_now > 0))
    print("进入主函数")
    #0.9->3.6 30分以下
    #50-> 3.69    34分左右
    result = ItemBasedFactory(train_matrix_now, result, average,20).get_result()
    print("出去主函数")
    try:
     np.savetxt("resource/result_item" + str(count) + ".csv", result, delimiter=',')
    except Exception as e:
        print(e)
    print("结果矩阵集导出")
    records_real = []
    records_predict = []
    for i in test_now.itertuples():
        try:
            ss = result[i[1], i[2]]
            records_predict.append(ss)
            records_real.append(getattr(i, "rating"))
        except:
            records_predict.append(0)
            records_real.append(getattr(i, "rating"))
    final_item_RMSE.append(get_rmse(records_real, records_predict))
    print("第%s次交叉测试item_rmse为%s" % (count, final_item_RMSE[count - 1]))
    print("ITEM_BASED结束")
    print("****************")


if __name__ == "__main__":
    # test_now = pd.read_csv("resource/test1.csv")
    # result = np.loadtxt("resource/result_item1.csv",delimiter=",")
    # #print(result)
    # records_predict=[]
    # records_real=[]
    # records_predict2 = []
    # for i in test_now.itertuples():
    #     try:
    #         ss = result[i[2], i[3]]
    #         if ss<10 is False:
    #             print()
    #         if np.isnan(ss):
    #              print(ss)
    #              records_predict2.append(0)
    #         else:
    #              records_predict2.append(ss)
    #         records_predict.append(ss)
    #         records_real.append(getattr(i, "rating"))
    #     except Exception as e:
    #         print(e)
    #         records_predict.append(0)
    #         records_real.append(getattr(i, "rating"))
    # #print(records_predict)
    # ss=get_rmse(records_real, records_predict)
    # #ss2=np.sqrt(((records_real - records_predict) ** 2).mean())
    # gg=get_rmse(records_real,records_predict2)
    # print(ss)
    # #print(ss2)
    # print(gg)
    print("测试开始")
    print("***********************************************************")
    print("***********************************************************")
    print("***********************************************************")
    print("***********************************************************")
    train_matrix = []
    test_ = []
    creat_train_matrix_and_test(train_matrix,test_,5)
    #get_train_matrix_and_test(train_matrix, test_, 5)
    count = 0
    final_user_RMSE = []
    final_item_RMSE = []
    for i in range(0, 5):
        train_matrix_now = train_matrix[i]
        test_now = test_[i]
        count = count + 1
        print("第%s次交叉测试开始:" % count)
        #item_rmse(count, train_matrix_now, test_now, final_item_RMSE)
        user_rmse(count, train_matrix_now, test_now, final_user_RMSE)
        print("第%s次交叉测试结束" % count)
        print("***********************************************************")
        print("***********************************************************")
        print("***********************************************************")
        print("***********************************************************")
    print("最后user_RMSE：")
    print(get_average(final_user_RMSE))
    print("最后item_RMSE：")
    print(get_average(final_item_RMSE))

