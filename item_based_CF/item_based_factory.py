import math
import numpy as np
from tqdm import tqdm

from item_based_CF.item_rating_compare import ItemRatingCompare


class ItemBasedFactory:
    def __init__(self, train_matrix, result, average,front):
        self.train_matrix = train_matrix
        self.front = front
        self.result = result
        self.matrix_sim = np.zeros((4125, 4125))
        self.average=average
        self.bool = True if len(str(front).split('.')) == 1 else False

    def _init(self):

        for item in tqdm(range(0, 4125)):
            try:
                ItemRatingCompare(self.train_matrix, item, self.matrix_sim).get_data()
            except:
                continue
            for user in range(0, 2967):

                if self.train_matrix[user][item] == 0:
                    try:
                        if self.bool:
                            front_index = np.argsort(-self.matrix_sim[item])[0:self.front]
                        else:
                            front_index = np.where(self.matrix_sim[item] > self.front)

                    except Exception as e:
                        print(e)
                        continue

                    ss = np.transpose(self.train_matrix)
                    nn = np.where(self.train_matrix[user, front_index] > 0)
                    if self.bool:
                        front_index = front_index[nn[0]]
                    else:
                        front_index = front_index[0][nn[1]]

                    if len(front_index) > 0:
                        try:
                         x = (np.sum(self.matrix_sim[user][front_index] * self.train_matrix[user][front_index])) \
                            / (np.sum(abs(self.matrix_sim[user][front_index])))
                        except:
                            continue
                        if x>6:
                         self.result[user][item] = 5
                        elif np.isnan(x):
                            self.result[user][item] = self.average[item]
                        else:
                         self.result[user][item] = x
                    else:
                        self.result[user][item] = self.average[item]
        # 所有user
        # for item in range(0, 4125):
        #     print("*第%s个物品的相关计算：" % item)
        #     try:
        #         compare_to = ItemRatingCompare(self.train, item, self.return_front,self.item_ratings_all)
        #     except:
        #         continue
        #     front_n = compare_to.get_front_n()
        #     print("*相关性计算完成*")
        #     # 每个user的 所有item
        #     print("*物品喜爱程度计算开始*")
        #     for user in range(0, 2967):
        #         numerator = 0
        #         denominator = 0
        #         try:
        #             if self.train.loc[user, item] == 0:
        #                 # 每个user的每个item
        #                 for ItemAndCompare in front_n:
        #                     if self.train.loc[user, ItemAndCompare.item_id] != 0:
        #                         numerator = numerator + ItemAndCompare.sim * self.train.loc[
        #                             user, ItemAndCompare.item_id]
        #                         denominator = denominator + abs(ItemAndCompare.sim)
        #             if denominator==0:
        #                 continue
        #             self.result.loc[user, item] = numerator / denominator
        #         except Exception as e:
        #             continue
        #     print("*物品喜爱程度计算完成*")

    def get_result(self):
        self._init()
        return self.result
