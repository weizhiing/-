import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.setrecursionlimit(2500)


class CompareTo():
    def __init__(self, train, user_id, matrix_sim, users_average):

        self.train = train
        self.user_id = user_id
        self.matrix_sim = matrix_sim
        self.users_average = users_average

    def _init(self):
        # 获取当前用户对物品的rating
        # try:
        #     user_ratings = self.train.loc[self.user_id, self.train.loc[self.user_id] > 0]
        # except:
        #     self.users_average[self.user_id] = 0
        #     print("用户不存在")
        #     return
        # if len(user_ratings)==0:
        #     return

        for now_user_id in range(self.user_id, 2967):
            if now_user_id == self.user_id:
                self.matrix_sim[self.user_id][now_user_id] = 1
            else:
                con_index = np.nonzero(self.train[self.user_id] * self.train[now_user_id])
                # 无共同喜欢物品
                if len(con_index[0]) <= 1:
                    self.matrix_sim[self.user_id][now_user_id] = 0
                    self.matrix_sim[now_user_id][self.user_id] = 0
                    continue
                # 相似度计算
                try:
                    self.matrix_sim[self.user_id][now_user_id] = np.sum(
                        (self.train[self.user_id][con_index] - self.users_average[self.user_id]) * (
                                self.train[now_user_id][con_index] - self.users_average[now_user_id])) / (np.sqrt(
                        np.sum(np.square(
                            self.train[self.user_id][con_index] - self.users_average[self.user_id]))) * np.sqrt(
                        np.sum(np.square(self.train[now_user_id][con_index] - self.users_average[now_user_id]))))
                except:
                    continue
                if self.matrix_sim[self.user_id][now_user_id] <= 0:
                    self.matrix_sim[self.user_id][now_user_id] = 0
                if self.matrix_sim[self.user_id][now_user_id] > 2:
                    [self.user_id][now_user_id] = 1.5
                if np.isnan(self.matrix_sim[self.user_id][now_user_id]):
                    self.matrix_sim[self.user_id][now_user_id] =0
                self.matrix_sim[now_user_id][self.user_id] = self.matrix_sim[self.user_id][now_user_id]
                # if user != self.user_id:
            #     try:
            #         now_user_ratings = self.train.loc[user, self.train.loc[user] > 0]
            #     except:
            #         self.users_average[user] = 0
            #         continue
            #     if len(now_user_ratings)==0:
            #         continue
            #     # 开始计算相关性
            #     # 分子
            #     numerator = 0
            #     # 分母
            #     denominator_1 = 0
            #     denominator_2 = 0
            #     for key in user_ratings.keys():
            #         try:
            #             # print(user + "bb")
            #             a = (now_user_ratings.get(key) - now_user_ratings_average)
            #             b = (user_ratings.get(key) - self.user_ratings_average)
            #             numerator = numerator + a * b
            #             denominator_1 = denominator_1 + a * a
            #             denominator_2 = denominator_2 + b * b
            #         except:
            #             # print(user + "cc")
            #             a = 0
            #     denominator = (math.sqrt(denominator_1) * math.sqrt(denominator_2))
            #     if denominator != 0:
            #         sim = numerator / denominator
            #     else:
            #         sim = 0
            #
            #     if sim > 0:
            #         self.front_n.append(UserAndCompare(user_id=user, sim=sim))

    # def userSort(self):
    #     self.quickSort(self.front_n, 0, len(self.front_n) - 1)
    #     self.front_n.reverse()

    def get_data(self):
        self._init()
        # self.userSort()
        # if len(self.front_n) <= self.return_front:
        #     return self.front_n
        # return self.front_n[0:self.return_front]

    # def partition(self, arr, low, high):
    #     i = (low - 1)  # 最小元素索引
    #     pivot = arr[high].sim
    #
    #     for j in range(low, high):
    #         # 当前元素小于或等于 pivot
    #         if arr[j].sim <= pivot:
    #             i = i + 1
    #             arr[i], arr[j] = arr[j], arr[i]
    #     arr[i + 1], arr[high] = arr[high], arr[i + 1]
    #     return (i + 1)
    #
    #     # arr[] --> 排序数组
    #
    # # low  --> 起始索引
    # # high  --> 结束索引
    #
    # # 快速排序函数
    # def quickSort(self, arr, low, high):
    #     if low < high:
    #         pi = self.partition(arr, low, high)
    #         self.quickSort(arr, low, pi - 1)
    #         self.quickSort(arr, pi + 1, high)
