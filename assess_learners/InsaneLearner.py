import numpy as np
import LinRegLearner as lrl
import BagLearner as bl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = 20 * [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)]

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def author(self):
        return "bflores9"

    def query(self, points):
        predictions = []
        for learner in self.learners:
            predictions.append(learner.query(points))

        return np.mean(predictions)