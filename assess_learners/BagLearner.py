import numpy as np

class BagLearner:
    def __init__(self, learner, bags, kwargs=None, boost=False, verbose=False):
        self.learner = learner
        self.bags = bags
        self.kwargs = kwargs if kwargs is not None else {}
        self.boost = boost
        self.verbose = verbose
        self.learners = []

    def author(self):
        return "bflores9"

    def add_evidence(self, data_x, data_y):
        for _ in range(self.bags):
            indices = np.random.choice(data_x.shape[0], size=data_x.shape[0], replace=True)
            bag_x = data_x[indices]
            bag_y = data_y[indices]

            learner = self.learner(**self.kwargs)
            learner.add_evidence(bag_x, bag_y)

            self.learners.append(learner)

        if self.verbose:
            print(f"Trained {self.bags} learners")

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        avg = np.mean(predictions, axis=0)
        if self.verbose:
            print(f"Average: ${avg}")
        return avg
