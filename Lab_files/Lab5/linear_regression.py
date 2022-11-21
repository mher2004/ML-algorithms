import numpy as np
import progressbar

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

class MyLinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
    """
    This class implements linear regression models
    Params:
    --------
    regularization - None for no regularization
                    'l2' for ridge regression
                    'l1' for lasso regression

    lam - lambda parameter for regularization in case of 
        Lasso and Ridge

    learning_rate - learning rate for gradient descent algorithm, 
                    used in case of Lasso

    tol - tolerance level for weight change in gradient descent
    """
    
    self.regularization = regularization 
    self.lam = lam 
    self.learning_rate = learning_rate 
    self.tol = tol
    self.weights = None
  
  def fit(self, X, y):
    
    X = np.array(X)
    # first insert a column with all 1s in the beginning
    # hint: you can use the function np.insert
    # YOUR CODE HERE

    if self.regularization is None:
      # the case when we don't apply regularization
      self.weights = # YOUR CODE HERE
    elif self.regularization == 'l2':
      # the case of Ridge regression
      self.weights = # YOUR CODE HERE
    elif self.regularization == 'l1':
      # in case of Lasso regression we use gradient descent
      # to find the optimal combination of weights that minimize the 
      # objective function in this case (slide 37)
      
      # initialize random weights, for example normally distributed
      self.weights = # YOUR CODE HERE

      converged = False
      # we can store the loss values to see how fast the algorithm converges
      self.loss = []
      # just a counter of algorithm steps
      i = 0 
      while (not converged):
        i += 1
        # calculate the predictions in case of the weights in this stage
        y_pred = # YOUR CODE HERE
        # calculate the mean squared error (loss) for the predictions
        # obtained above
        self.loss.append('YOUR CODE HERE')
        # calculate the gradient of the objective function with respect to w
        # for the second component \sum|w_i| use np.sign(w_i) as it's derivative
        grad = # YOUR CODE HERE
        new_weights = self.weights - self.learning_rate * grad
        # check whether the weights have changed a lot after this iteration
        # compute the norm of difference between old and new weights 
        # and compare with the pre-defined tolerance level, if the norm
        # is smaller than the tolerance level then we consider convergence
        # of the algorithm
        converged = # YOUR CODE HERE
        self.weights = new_weights
      print(f'Converged in {i} steps')

  def predict(self, X):
    X = np.array(X)
    # don't forget to add the feature of 1s in the beginning
    X = # YOUR CODE HERE
    # predict using the obtained weights
    return # YOUR CODE HERE 