import numpy as np
import progressbar

# you need Regresssion trees, so use either your implementation or sklearn's
from decision_tree import RegressionTree 

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

# Instead of using a Decision Tree with one level
# we can create another object for Decision Stump
# which will work faster since it will not compute impurity
# to decide on which feature to make a split

# after implementing this version, create a different Adaboost
# that uses decision trees with one level and check that it is 
# more inefficient compared to the below implementation.

class DecisionStump():
  def __init__(self):
    # we will use this attribute to convert the predictions
    # in case the error > 50%
    self.flip = 1
    # the feature index on which the split was made
    self.feature_index = None
    # the threshold based on which the split was made
    self.threshold = None
    # the confidence of the model (see the pseudocode from the lecture slides)
    self.alpha = None

class Adaboost():
  # this implementation supports only -1,1 label encoding
  def __init__(self, nr_estimators=5):
    # number of weak learners (Decision Stumps) to use
    self.nr_estimators = nr_estimators
    self.progressbar = progressbar.ProgressBar(widgets=widgets)

  def fit(self, X, y):
    X = X.values
    y = y.values
    nr_samples, nr_features = np.shape(X)

    # initialize the uniform weights for each training instance
    # YOUR CODE HERE
    
    self.models = []
    for i in self.progressbar(range(self.nr_estimators)):
        model = DecisionStump()

        # we set the initial error very high in order to select 
        # the model with lower error 
        min_error = 1 

        # we go over each feature as in case of decision tree
        # to decide which split leads to a smaller error
        # note that here we don't care about the impurity
        # even if we find a model with 90% error, we will flip the
        # sign of the predictions and will make it a model with 10% error
        for feature_id in range(nr_features):
          unique_values = np.unique(X[:, feature_id])
          thresholds = (unique_values[1:] + unique_values[:-1]) / 2
          for threshold in thresholds:
              # setting an intial value for the flip
              flip = 1
              # setting all the predictions as 1
              prediction = np.ones(nr_samples)
              # if the feature has values less than the fixed threshold
              # then it's prediction should be manually put as -1
              prediction['YOUR CODE HERE'] = -1

              # compute the weighted error (epsilon_t) for the resulting prediction
              error = # YOUR CODE HERE
              
              # if the model is worse than random guessing
              # then we need to set the flip variable to -1 
              # so that we can use it later, we also modify the error
              # accordingly
              if error > 0.5:
                error = 1 - error
                flip = -1

              # if this feature and threshold were the one giving 
              # the smallest error, then we store it's info in the 'model' object
              if error < min_error:
                model.flip = flip
                model.threshold = threshold
                model.feature_index = feature_id
                min_error = error
        
        # compute alpha based on the error of the 'best' decision stump
        model.alpha = # YOUR CODE HERE

        # obtain the predictions from the chosen decision stump
        # using the info stored in the 'model' object
        # don't forget about the flip if necessary
        # YOUR CODE HERE

        # compute the weights and normalize them
        w *= # YOUR CODE HERE 
        w /= np.sum(w)

        # store the decision stump of the current iteration for later
        self.models.append(model)

  def predict(self, X):
    X = X.values
    nr_samples = np.shape(X)[0]
    y_pred = np.zeros(nr_samples)

    # for each instance in X you should obtain the 'prediction'
    # from each decision stump (not forgetting about the flip variable)
    # then take the sum of 
    # all the individual predictions times their weights (alpha)
    # if the resulting amount is bigger than 0 then predict 1, otherwise -1
    # YOUR CODE HERE
    return y_pred


class GradientBoostingRegressor:
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                min_impurity=1e-7, max_depth=4):      
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.bar = progressbar.ProgressBar(widgets=widgets)
    
    # write the square loss function as in the lectures
    def square_loss(y, y_pred): return # YOUR CODE HERE

    # write the gradient of square loss as in the lectures
    def square_loss_gradient(y, y_pred): return # YOUR CODE HERE

    self.loss = square_loss
    self.loss_gradient = square_loss_gradient

  def fit(self, X, y):
    self.trees = [] # we will store the regression trees per iteration
    self.train_loss = [] # we will store the loss values per iteration

    # initialize the predictions (f(x) in the lectures) 
    # with the mean values of y
    # hint: you may want to use the np.full function here
    self.mean_y = np.mean(y)
    y_pred = # YOUR CODE HERE
    for i in self.bar(range(self.n_estimators)):     
      tree = RegressionTree(
              min_samples_split=self.min_samples_split,
              min_impurity=self.min_impurity,
              max_depth=self.max_depth) # this is h(x) from our lectures
      # get the loss when comparing y_pred with true y
      # and store the values in self.train_loss
      # YOUR CODE HERE

      # get the pseudo residuals
      residuals = # YOUR CODE HERE 

      tree.fit(X, residuals) # fit the tree on the residuals
      # update the predictions y_pred using the tree predictions on X
      # YOUR CODE HERE

      self.trees.append(tree) # stor the tree model

  def predict(self, X):
    # start with initial predictions as vector of 
    # the mean values of y_train (self.mean_y)
    y_pred = # YOUR CODE HERE
    # iterate over the regression trees and apply the same gradient updates
    # as in the fitting process, but using test instances
    for tree in self.trees:
        y_pred -= # YOUR CODE HERE
    return y_pred


class GradientBoostingRegressor:
  def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                min_impurity=1e-7, max_depth=4):      
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.bar = progressbar.ProgressBar(widgets=widgets)
    # YOUR CODE HERE
    
  def fit(self, X, y):
    # YOUR CODE HERE

  def predict(self, X):    
    # YOUR CODE HERE
