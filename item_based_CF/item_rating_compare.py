import sys

import math
import numpy as np

sys.setrecursionlimit(2500)


class ItemRatingCompare:
    def __init__(self, train_matrix, item_id, matrix_sim):

        self.train_matrix = np.transpose(train_matrix)
        self.item_id = item_id
        self.matrix_sim = matrix_sim

    def _init(self):
        for now_item_id in range(self.item_id, 4125):
            if now_item_id == self.item_id:
                self.matrix_sim[self.item_id][now_item_id] = 1
            else:
                con_index = np.nonzero(self.train_matrix[self.item_id] * self.train_matrix[now_item_id])
                # 无共同喜欢物品
                if len(con_index[0]) <= 1:
                    self.matrix_sim[self.item_id][now_item_id] = 0
                    self.matrix_sim[self.item_id][now_item_id] = 0
                    continue
                # 相似度计算
                try:
                 self.matrix_sim[self.item_id][now_item_id] = np.sum(
                    self.train_matrix[self.item_id, con_index] * self.train_matrix[now_item_id, con_index]) / \
                    (np.sqrt(np.sum(np.square(self.train_matrix[self.item_id, con_index]))) *
                     np.sqrt(np.sum(np.square(self.train_matrix[now_item_id, con_index]))))
                except:
                    continue
                if self.matrix_sim[self.item_id][now_item_id] <= 0:
                    self.matrix_sim[self.item_id][now_item_id] = 0
                if np.isnan(self.matrix_sim[self.item_id][now_item_id]):
                    self.matrix_sim[self.item_id][now_item_id] = 0
                self.matrix_sim[now_item_id][self.item_id] = self.matrix_sim[self.item_id][now_item_id]

        # 实验1 均为0 是否会loc 出错
        # 实验二 下是否可行
        # ratings 的格式
        # try:
        #     item_ratings = self.train_matrix.loc[self.train_matrix.loc[:, self.item_id] > 0, self.item_id]
        # except Exception as e:
        #     print("物品不存在")
        #     return
        # if len(item_ratings)==0 :
        #     return
        #
        # for item in range(0, 4125):
        #
        #     if item != self.item_id:
        #
        #         try:
        #             now_item_ratings = self.train_matrix.loc[(self.train_matrix.loc[:, item] > 0)&
        #                                                      (self.train_matrix.loc[:, self.item_id] > 0), item]
        #         except Exception as e :
        #             continue
        #         if now_item_ratings is None:
        #             continue
        #         # 开始计算相关性
        #         # 分子
        #         numerator = 0
        #         # 分母
        #         denominator_1 = 0
        #         denominator_2 = 0
        #         for key in now_item_ratings.keys():
        #                 a = item_ratings.get(key)
        #                 b = now_item_ratings.get(key)
        #                 numerator = numerator + a * b
        #                 denominator_1 = denominator_1 + a * a
        #                 denominator_2 = denominator_2 + b * b
        #         denominator = (math.sqrt(denominator_1) * math.sqrt(denominator_2))
        #         if denominator != 0:
        #             sim = numerator / denominator
        #         else:
        #             sim = 0
        #
        #         if sim > 0:
        #             self.front_n.append(ItemAndCompare(item_id=item, sim=sim))

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
