import numpy as np

from user_based_CF.compare_to import CompareTo
import math
from tqdm import tqdm


class UserBasedFactory:
    def __init__(self, train_matrix, result, users_average, front):
        self.train_matrix = train_matrix
        self.matrix_sim = np.zeros((2967, 2967))
        self.result = result
        self.front = front
        self.users_average = users_average
        self.bool = True if len(str(front).split('.')) == 1 else False

    def _init(self):
        # 所有user
        for user in tqdm(range(0, 2967)):
            # print("*第%s个用户的相关计算：" % user)
            try:
                CompareTo(self.train_matrix, user, self.matrix_sim, self.users_average).get_data()
            except:
                continue
            # front_n = compare_to.get_data()
            # average = self.users_average[user]
            # print("*相关性计算完成*")
            # 每个user的 所有item
            # print("*物品喜爱程度计算开始*")
            for item in range(0, 4125):
                # print(self.matrix_sim[user])
                if self.train_matrix[user][item] == 0:
                    try:
                        # 相关矩阵中 相关性大于n的用户id
                        if self.bool:
                            front_index = np.argsort(-self.matrix_sim[user])[0:self.front]
                        else:
                            front_index = np.where(self.matrix_sim[user] > self.front)


                    except Exception as e:
                        print(e)
                        continue

                    x = self.users_average[user]
                    nn = np.where(self.train_matrix[front_index, item] > 0)
                    if self.bool:
                        front_index = front_index[nn[0]]
                    else:
                        front_index = front_index[0][nn[1]]
                    if len(front_index) > 0:

                        ss = np.transpose(self.train_matrix)
                        # front_index选取相似度大于一定的，nn是从相似度一定中选item有评分的，最后赋值到front——index


                        try:
                            x = x + \
                                (np.sum(self.matrix_sim[user][front_index] * (ss[item][front_index] -
                                                                              self.users_average[front_index]))) \
                                / (np.sum(abs(self.matrix_sim[user][front_index])))
                        except:
                            self.result[user][item] = x
                            continue
                        if x>6:
                           self.result[user][item] = 5
                        elif np.isnan(x):
                            self.result[user][item] = self.users_average[user]
                        else:
                            self.result[user][item] = x
                    else:
                        self.result[user][item] = x
                    # numerator = 0
                    # denominator = 0
                    # try:
                    #     if self.train.loc[user, item] == 0:
                    #         # 每个user的每个item
                    #         for userAndCompare in front_n:
                    #             if self.train.loc[userAndCompare.user_id, item] != 0:
                    #                 numerator = numerator + userAndCompare.sim * (
                    #                         self.train.loc[userAndCompare.user_id, item] - self.users_average[
                    #                     userAndCompare.user_id])
                    #                 denominator = denominator + abs(userAndCompare.sim)
                    #     if denominator != 0:
                    #         x = x + numerator / denominator
                    #     self.result.loc[user, item] = x
                    # except:
                    #     continue
            # print("*物品喜爱程度计算完成*")

    def get_result(self):
        self._init()
        return self.result
