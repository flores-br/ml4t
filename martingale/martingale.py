""""""
import numpy

"""Assess a betting strategy.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Bryan Flores  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: bflores9		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903848430 		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import numpy as np  		  	   		  		 			  		 			 	 	 		 		 	
import matplotlib.pyplot as plt
  		  	   		  		 			  		 			 	 	 		 		 	
def author():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    return "bflores9"  # replace tb34 with your Georgia Tech username.
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
def gtid():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    return 903848430  # replace with your GT ID number
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
def get_spin_result(win_prob):  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param win_prob: The probability of winning  		  	   		  		 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    result = False  		  	   		  		 			  		 			 	 	 		 		 	
    if np.random.random() <= win_prob:  		  	   		  		 			  		 			 	 	 		 		 	
        result = True  		  	   		  		 			  		 			 	 	 		 		 	
    return result

def gamble_sim(win_prob):
    episode_winnings = 0
    episode_arr = np.full(1001, 80)
    episode_arr[0] = 0 # initial value
    counter = 1
    while episode_winnings < 80 and counter < 1000:
        won = False
        bet_amount = 1

        while not won:
            won = get_spin_result(win_prob)

            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount *= 2

            episode_arr[counter] = episode_winnings
            counter += 1

    return episode_arr

def gamble_sim_bankroll(win_prob, bankroll=256):
    episode_winnings = 0
    episode_arr = np.full(1001, 80)
    episode_arr[0] = 0 # initial value
    counter = 1
    while -bankroll < episode_winnings < 80 and counter < 1000:
        won = False
        bet_amount = 1

        while not won:
            won = get_spin_result(win_prob)

            if won:
                episode_winnings += bet_amount
            else:
                episode_winnings -= bet_amount
                bet_amount = bankroll + episode_winnings if episode_winnings - bet_amount < -bankroll else bet_amount * 2

            episode_arr[counter] = episode_winnings
            counter += 1

    # if no more money from bankroll
    if episode_winnings == -bankroll:
        episode_arr[counter:] = -bankroll

    return episode_arr

# ================== EXPERIMENT 1 ==================
def figure_1(win_prob):
    # plot definitions
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 1 - Winnings from 10 episodes")
    plt.xlabel("Trials (n)")
    plt.ylabel("Winnings ($)")

    for _ in range(10):
        curr_episode = gamble_sim(win_prob)
        plt.plot(curr_episode)

    plt.savefig("figure1.png")
    plt.clf()

def figure_2(win_prob):
    # plot definitions
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 2 - Mean and standard deviations from 1000 episodes")
    plt.xlabel("Trials (n)")
    plt.ylabel("Winnings ($)")

    arr = np.zeros((1000, 1001))

    for i in range(1000):
        arr[i] = gamble_sim(win_prob)

    mean = numpy.mean(arr, axis=0)
    std = numpy.std(arr, axis=0)

    # plot
    plt.plot(mean, label="Mean")
    plt.plot(mean + std, label="Mean + std")
    plt.plot(mean - std, label="Mean - std")
    plt.legend()
    plt.savefig("figure2.png")
    plt.clf()

def figure_3(win_prob):
    # plot definitions
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 3 - Median and standard deviations from 1000 episodes")
    plt.xlabel("Trials (n)")
    plt.ylabel("Winnings ($)")

    arr = np.zeros((1000, 1001))

    for i in range(1000):
        arr[i] = gamble_sim(win_prob)

    median = numpy.median(arr, axis=0)
    std = numpy.std(arr, axis=0)

    # plot
    plt.plot(median, label="Median")
    plt.plot(median + std, label="Median + std")
    plt.plot(median - std, label="Median - std")
    plt.legend()
    plt.savefig("figure3.png")
    plt.clf()

# ================== EXPERIMENT 2 ==================
def figure_4(win_prob):
    # plot definitions
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 4 - Means from 1000 episodes using a bankroll of 256$")
    plt.xlabel("Trials (n)")
    plt.ylabel("Winnings ($)")

    arr = np.zeros((1000, 1001))

    for i in range(1000):
        arr[i] = gamble_sim_bankroll(win_prob)

    mean = numpy.mean(arr, axis=0)
    std = numpy.std(arr, axis=0)

    # plot
    plt.plot(mean, label="Mean")
    plt.plot(mean + std, label="Mean + std")
    plt.plot(mean - std, label="Mean - std")
    plt.legend()
    plt.savefig("figure4.png")
    plt.clf()

def figure_5(win_prob):
    # plot definitions
    plt.axis([0, 300, -256, 100])
    plt.title("Figure 5 - Medians from 1000 episodes using a bankroll of 256$")
    plt.xlabel("Trials (n)")
    plt.ylabel("Winnings ($)")

    arr = np.zeros((1000, 1001))

    for i in range(1000):
        arr[i] = gamble_sim_bankroll(win_prob)

    median = numpy.median(arr, axis=0)
    std = numpy.std(arr, axis=0)

    # plot
    plt.plot(median, label="Mean")
    plt.plot(median + std, label="Mean + std")
    plt.plot(median - std, label="Mean - std")
    plt.legend()
    plt.savefig("figure5.png")
    plt.clf()

def test_code():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Method to test your code  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    win_prob = 9 / 19  # p(black) = 1 / (1 + 10/9)
    np.random.seed(gtid())  # do this only once  		  	   		  		 			  		 			 	 	 		 		 	
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		  		 			  		 			 	 	 		 		 	
    # add your code here to implement the experiments
    figure_5(win_prob)
  		  	   		  		 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    test_code()  		  	   		  		 			  		 			 	 	 		 		 	
