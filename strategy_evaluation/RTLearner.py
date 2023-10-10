import numpy as np
from scipy import stats

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print("DTLearner")
            print(self.tree)

    def author(self):
        return 'bflores9'

    def query(self, points):
        predictions = np.empty(points.shape[0])
        for i in range(points.shape[0]):
            predictions[i] = self.get_prediction(points[i])
        return predictions

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size:
            return np.array([np.nan, stats.mode(data_y)[0][0], np.nan, np.nan])

        if np.all(data_y == data_y[0]):
            return np.array([np.nan, data_y[0], np.nan, np.nan])

        # find random feature
        rand_feature = np.random.randint(0, data_x.shape[1])
        split_val = np.median(data_x[:, rand_feature])

        left_indices = data_x[:, rand_feature] <= split_val

        if np.all(np.isclose(left_indices, left_indices[0])):
            return np.array([np.nan, stats.mode(data_y)[0][0], np.nan, np.nan])

        right_indices = np.logical_not(left_indices)

        left_tree = self.build_tree(data_x[left_indices], data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices], data_y[right_indices])

        if left_tree.ndim == 1:
            root = np.array([rand_feature, split_val, 1, 2])
        else:
            root = np.array([rand_feature, split_val, 1, left_tree.shape[0] + 1])

        return np.vstack((root, left_tree, right_tree))

    def get_prediction(self, point):
        cur = 0
        while np.logical_not(np.isnan(self.tree[cur][0])):
            split_value = point[int(self.tree[cur][0])]

            if split_value <= self.tree[cur][1]:
                cur += int(self.tree[cur][2])
            else:
                cur += int(self.tree[cur][3])
        return self.tree[cur][1]

    def author(self):
        return "bflores9"
