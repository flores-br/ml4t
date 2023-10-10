""""""  		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
"""  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import math  		  	   		  		 			  		 			 	 	 		 		 	
import sys  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import numpy as np
import matplotlib.pyplot as plt
  		  	   		  		 			  		 			 	 	 		 		 	
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

def question_1(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 101, 1)
    train_errors = np.zeros(leaf_sizes.size)
    test_errors = np.zeros(leaf_sizes.size)

    for leaf_size in leaf_sizes:
        model = dt.DTLearner(leaf_size=leaf_size)
        model.add_evidence(train_x, train_y)

        # train
        train_predictions = model.query(train_x)
        train_errors[leaf_size - 1] = np.sqrt(np.mean((train_y - train_predictions) ** 2))

        # test
        test_predictions = model.query(test_x)
        test_errors[leaf_size - 1] = np.sqrt(np.mean((test_y - test_predictions) ** 2))

    plt.plot(leaf_sizes, train_errors, label='Training Error')
    plt.plot(leaf_sizes, test_errors, label='Testing Error')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Figure 1 - DTLearner: Training/Testing Error vs. Leaf Size')
    plt.grid()
    plt.legend()
    plt.savefig("figure_1.png")
    plt.clf()

    plt.plot(leaf_sizes[10:-10], train_errors[10:-10], label='Training Error')
    plt.plot(leaf_sizes[10:-10], test_errors[10:-10], label='Testing Error')
    plt.xlabel('Leaf Size')
    plt.xticks(np.arange(10, 91, 5))
    plt.ylabel('RMSE')
    plt.title('Figure 2 - DTLearner: Training/Testing Error vs. Leaf Size (Zoomed)')
    plt.grid()
    plt.legend()
    plt.savefig("figure_2.png")
    plt.clf()

def question_2(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 101, 1)
    train_errors = np.zeros(leaf_sizes.size)
    test_errors = np.zeros(leaf_sizes.size)
    bags = 10

    for leaf_size in leaf_sizes:
        model = bl.BagLearner(learner=dt.DTLearner, bags=bags, kwargs={'leaf_size': leaf_size}, boost=False, verbose=False)
        model.add_evidence(train_x, train_y)

        # train
        train_predictions = model.query(train_x)
        train_errors[leaf_size - 1] = np.sqrt(np.mean((train_y - train_predictions) ** 2))

        # test
        test_predictions = model.query(test_x)
        test_errors[leaf_size - 1] = np.sqrt(np.mean((test_y - test_predictions) ** 2))

    plt.plot(leaf_sizes, train_errors, label='Training Error')
    plt.plot(leaf_sizes, test_errors, label='Testing Error')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title(f'Figure 3 - BagLearner ({bags} bags) w/ DTLearner: Training/Testing Error vs. Leaf Size')
    plt.grid()
    plt.legend()
    plt.savefig("figure_3.png")
    plt.clf()

def question_3_1(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 101, 1)
    dt_mae = np.zeros(leaf_sizes.size)
    rt_mae = np.zeros(leaf_sizes.size)

    for leaf_size in leaf_sizes:
        # DTLearner
        dt_learner = dt.DTLearner(leaf_size=leaf_size)
        dt_learner.add_evidence(train_x, train_y)
        dt_predictions = dt_learner.query(test_x)
        dt_mae[leaf_size - 1] = np.mean(np.abs(test_y - dt_predictions))

        # RTLearner
        rt_learner = rt.RTLearner(leaf_size=leaf_size)
        rt_learner.add_evidence(train_x, train_y)
        rt_predictions = rt_learner.query(test_x)
        rt_mae[leaf_size - 1] = np.mean(np.abs(test_y - rt_predictions))

    plt.plot(leaf_sizes, dt_mae, label='DTLearner')
    plt.plot(leaf_sizes, rt_mae, label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Figure 4 - DTLearner vs. RTLearner (MAE)')
    plt.grid()
    plt.legend()
    plt.savefig("figure_4.png")
    plt.clf()

def calculate_r_squared(actual, predicted):
    mean_actual = np.mean(actual)
    ss_total = np.sum((actual - mean_actual) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def question_3_2(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 101, 1)
    dt_rs = np.zeros(leaf_sizes.size)
    rt_rs = np.zeros(leaf_sizes.size)

    for leaf_size in leaf_sizes:
        # DTLearner
        dt_learner = dt.DTLearner(leaf_size=leaf_size)
        dt_learner.add_evidence(train_x, train_y)
        dt_predictions = dt_learner.query(test_x)
        dt_rs[leaf_size - 1] = calculate_r_squared(test_y, dt_predictions)

        # RTLearner
        rt_learner = rt.RTLearner(leaf_size=leaf_size)
        rt_learner.add_evidence(train_x, train_y)
        rt_predictions = rt_learner.query(test_x)
        rt_rs[leaf_size - 1] = calculate_r_squared(test_y, rt_predictions)

    plt.plot(leaf_sizes, dt_rs, label='DTLearner')
    plt.plot(leaf_sizes, rt_rs, label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('R-Squared')
    plt.title('Figure 4 - DTLearner vs. RTLearner (R-Squared)')
    plt.grid()
    plt.legend()
    plt.savefig("figure_5.png")
    plt.clf()

if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    if len(sys.argv) != 2:  		  	   		  		 			  		 			 	 	 		 		 	
        print("Usage: python testlearner.py <filename>")  		  	   		  		 			  		 			 	 	 		 		 	
        sys.exit(1)  		  	   		  		 			  		 			 	 	 		 		 	
    inf = open(sys.argv[1])  		  	   		  		 			  		 			 	 	 		 		 	
    data = np.array([list(map(str,s.strip().split(','))) for s in inf.readlines()])

    if sys.argv[1] == "Data/Istanbul.csv":
        data = data[1:, 1:]

    data = data.astype('float')

    # compute how much of the data is training and testing  		  	   		  		 			  		 			 	 	 		 		 	
    train_rows = int(0.6 * data.shape[0])  		  	   		  		 			  		 			 	 	 		 		 	
    test_rows = data.shape[0] - train_rows  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    # separate out training and testing data  		  	   		  		 			  		 			 	 	 		 		 	
    train_x = data[:train_rows, 0:-1]  		  	   		  		 			  		 			 	 	 		 		 	
    train_y = data[:train_rows, -1]  		  	   		  		 			  		 			 	 	 		 		 	
    test_x = data[train_rows:, 0:-1]  		  	   		  		 			  		 			 	 	 		 		 	
    test_y = data[train_rows:, -1]

    learner = dt.DTLearner()
    learner.add_evidence(train_x, train_y)
  		  	   		  		 			  		 			 	 	 		 		 	
    print(f"{test_x.shape}")  		  	   		  		 			  		 			 	 	 		 		 	
    print(f"{test_y.shape}")

    question_1(train_x, train_y, test_x, test_y)
  		  	   		  		 			  		 			 	 	 		 		 	
    # create a learner and train it  		  	   		  		 			  		 			 	 	 		 		 	
    # learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = 10, boost = False, verbose = True)
    # learner.add_evidence(train_x, train_y)  # train it
    # Y = learner.query(test_x)
    # print(learner.author())
  		  	   		  		 			  		 			 	 	 		 		 	
    # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
    #
    # # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
