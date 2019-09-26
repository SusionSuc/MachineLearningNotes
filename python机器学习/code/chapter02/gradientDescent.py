import numpy as pd
import matplotlib.pyplot as plt


class AdalineSGD(object):
    """
       Parameters
       ------------
       eta : float
         Learning rate (between 0.0 and 1.0)
       n_iter : int
         Passes over the training dataset.
       shuffle : bool (default: True)
         Shuffles training data every epoch if True to prevent cycles.
       random_state : int
         Random number generator seed for random weight
         initialization.
       Attributes
       -----------
       w_ : 1d-array
         Weights after fitting.
       cost_ : list
         Sum-of-squares cost function value averaged over all
         training samples in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.
        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X,y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
            return self