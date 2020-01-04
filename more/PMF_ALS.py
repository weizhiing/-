import numpy as np


class PMF:
    def __init__(self, item, user, train_data_matrix):
        self.item = item
        self.user = user
        self.train = train_data_matrix

    def get_data(self):
        while 1 == 1:
            new_user = self.updateUser(1, self.item, self.user, self.train)
            new_item = self.updateItem(1, new_user, self.item, self.train)
            if self.get_rmse(new_user, self.user) < 0.01 and self.get_rmse(new_item, self.item) < 0.01:
                break
            self.user = new_user
            self.item = new_item
        return np.dot(self.user.T, self.item)

    def get_rmse(self, o, train):
        add = 0.0
        row = o.shape[0]
        col = o.shape[1]
        a = row * col
        for i in range(row):
            for j in range(col):
                add += np.square(o[i][j] - train[i][j])
        return np.sqrt(add / a)

    def updateUser(self, number, item, user, train_data_matrix):
        N = user.shape[1]
        K = user.shape[0]
        trans_user = np.transpose(user)
        trans_user_copy = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                trans_user_copy[i][j] = trans_user[i][j]
        trans_item = np.transpose(item)
        for i in range(N):

            index = np.nonzero(train_data_matrix[i])
            ii_sum = np.zeros((K, K))
            ix_sum = np.zeros((K, 1))
            for j in index[0]:
                ii_sum += np.array([trans_item[j]]).T * np.array([trans_item[j]])

            for z in index[0]:
                ix_sum += np.array([trans_item[z]]).T * train_data_matrix[i][z]
            # 点积/逆矩阵/
            trans_user_copy[i] = np.array(np.dot(np.linalg.inv(number * np.identity(K) + ii_sum), ix_sum).T).reshape(K)
        return trans_user_copy.T

    def updateItem(self, number, user,item, train_data_matrix):
        train_data_matrix_tran = np.transpose(train_data_matrix)
        M = item.shape[1]
        K = item.shape[0]
        trans_user = np.transpose(user)
        trans_item = np.transpose(item)
        trans_item_copy = np.zeros((M, K))
        for i in range(M):
            for j in range(K):
                trans_item_copy[i][j] = trans_item[i][j]
        for i in range(M):

            index = np.nonzero(train_data_matrix_tran[i])
            uu_sum = np.zeros((K, K))
            ux_sum = np.zeros((K, 1))
            for j in index[0]:
                uu_sum += np.array([trans_user[j]]).T * np.array([trans_user[j]])

            for z in index[0]:
                ux_sum += np.array([trans_user[z]]).T * train_data_matrix_tran[i][z]

            trans_item_copy[i] = np.array(np.dot(np.linalg.inv(number * np.identity(K) + uu_sum), ux_sum).T).reshape(K)
        return trans_item_copy.T
