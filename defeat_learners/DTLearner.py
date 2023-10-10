""""""  		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import warnings  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import numpy as np  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
class DTLearner(object):  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		  		 			  		 			 	 	 		 		 	
    your own correct DTLearner from Project 3.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		  		 			  		 			 	 	 		 		 	
    :type leaf_size: int  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	

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
            return np.array([np.nan, np.mean(data_y), np.nan, np.nan])

        if np.all(data_y == data_y[0]):
            return np.array([np.nan, data_y[0], np.nan, np.nan])

        # find best feature
        max_corr = -1
        best_feature = None
        for i in range(data_x.shape[1]):
            cur_corr = np.abs(np.corrcoef(data_x[:, i], data_y)[0, 1])
            if cur_corr > max_corr:
                max_corr = cur_corr
                best_feature = i
        split_val = np.median(data_x[:, best_feature])

        left_indices = data_x[:, best_feature] <= split_val

        if np.all(np.isclose(left_indices, left_indices[0])):
            return np.array([np.nan, np.mean(data_y), np.nan, np.nan])

        right_indices = np.logical_not(left_indices)

        left_tree = self.build_tree(data_x[left_indices], data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices], data_y[right_indices])

        if left_tree.ndim == 1:
            root = np.array([best_feature, split_val, 1, 2])
        else:
            root = np.array([best_feature, split_val, 1, left_tree.shape[0] + 1])

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
