import numpy as np
from linear_model import LinearModel

class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.no_examples = x.shape[0]
        self.no_features = x.shape[1]
        self.x = x
        self.y = y
        

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            y_preds: Predicted y values of shape (m,).
        """

        y_preds = []

        #Iterate over validation set
        for i, x_p in enumerate(x):
            #Calculate weight matrix, which is diagonal
            W = np.zeros((self.no_examples, self.no_examples))

            #Iterate over training set
            for j, x_i in enumerate(self.x):
                arg = (np.linalg.norm(x_i - x_p))**2
                W[j][j] = np.exp(-arg / (2*self.tau**2))

            #Closed-form solution for theta
            self.theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y

            y_pred = np.dot(x_p, self.theta)
            y_preds.append(y_pred)

        return y_preds