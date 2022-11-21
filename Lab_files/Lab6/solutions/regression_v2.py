import numpy as np
import pandas as pd
import cvxpy
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
    y = np.array(y)
    # first insert a column with all 1s in the beginning
    # hint: you can use the function np.insert
    X = np.insert(X, 0, 1, axis=1)

    if self.regularization is None:
      # the case when we don't apply regularization
      self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
    elif self.regularization == 'l2':
      # the case of Ridge regression
      self.weights = np.linalg.inv(X.T @ X + np.diag([self.lam]*len(X.T))) @ X.T @ y
    elif self.regularization == 'l1':
      # in case of Lasso regression we use gradient descent
      # to find the optimal combination of weights that minimize the 
      # objective function in this case (slide 37)
      
      # initialize random weights, for example normally distributed
      self.weights = np.random.random(X.shape[1])

      converged = False
      # we can store the loss values to see how fast the algorithm converges
      self.loss = []
      # just a counter of algorithm steps
      i = 0
      while (not converged):
        i += 1
        # calculate the predictions in case of the weights in this stage
        y_pred = X @ self.weights
        # calculate the mean squared error (loss) for the predictions
        # obtained above
        self.loss.append(np.sum((y_pred - y)**2))
        # calculate the gradient of the objective function with respect to w
        # for the second component \sum|w_i| use np.sign(w_i) as it's derivative
        grad = - 2 * X.T @ (y - y_pred) + np.sum(np.sign(self.weights))
        new_weights = self.weights - self.learning_rate * grad
        # check whether the weights have changed a lot after this iteration
        # compute the norm of difference between old and new weights 
        # and compare with the pre-defined tolerance level, if the norm
        # is smaller than the tolerance level then we consider convergence
        # of the algorithm
        # print(np.linalg.norm(new_weights - self.weights))
        converged = np.linalg.norm(new_weights - self.weights) < self.tol
        self.weights = new_weights
      print(f'Converged in {i} steps')
    
  def predict(self, X):
    X = np.array(X)
    # don't forget to add the feature of 1s in the beginning
    X = np.insert(X, 0, 1, axis=1)
    # predict using the obtained weights
    return X @ self.weights
  


class DecisionNode:
  def __init__(self, feature_id=None, threshold=None,
                value=None, true_branch=None, false_branch=None):
      self.feature_id = feature_id          
      self.threshold = threshold          
      self.value = value                  
      self.true_branch = true_branch      
      self.false_branch = false_branch    

class RegressionTree:
  def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
    # YOUR CODE HERE
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self.root = self.grow_tree(X, y)

  def grow_tree(self, X, y, current_depth=0):
    largest_purity_gain = 0 # initial small value for purity gain
    nr_samples, nr_features = np.shape(X)

    # checking if we have reached the pre-specified limits
    if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:
      for feature_id in range(nr_features):
        for threshold in X[:, feature_id]:
          if isinstance(threshold, int) or isinstance(threshold, float):
            true_indices = X[:, feature_id] >= threshold
          else:
            true_indices = X[:, feature_id] == threshold

          X1, y1 = X[true_indices], y[true_indices]
          X2, y2 = X[~true_indices], y[~true_indices]                  

          # checking if we have samples in each subtree
          if len(X1)*len(X2) > 0:
            purity_gain = y.var() - (len(y1)*y1.var() + len(y2)*y2.var())/len(y)
            if purity_gain > largest_purity_gain:
              largest_purity_gain = purity_gain
              best_feature_id = feature_id
              best_threshold = threshold
              best_X1 = X1 # true
              best_y1 = y1 # true
              best_X2 = X2 # false
              best_y2 = y2 # false

      if largest_purity_gain > self.min_impurity:
        true_branch = self.grow_tree(best_X1, best_y1, current_depth + 1)
        false_branch = self.grow_tree(best_X2, best_y2, current_depth + 1)

        return DecisionNode(feature_id=best_feature_id,
                            threshold=best_threshold,
                            true_branch=true_branch,
                            false_branch=false_branch)

    return DecisionNode(value=y.mean())
   
  def predict(self, X):
    # YOUR CODE HERE
    X = np.array(X)
    y_pred = []
    for x in X:
      node = self.root
      while node.value is None:
        new_node = node.false_branch
        feature_value = x[node.feature_id]
        if isinstance(feature_value, int) or isinstance(feature_value, float):
          if feature_value >= node.threshold:
            new_node = node.true_branch
        elif feature_value == node.threshold:
          new_node = node.true_branch
        node = new_node
      y_pred.append(node.value)
    return y_pred

class SVR:
  def __init__(self, epsilon=0.1, C=1, kernel_name='linear', power=2, gamma=None, coef=2, eps=1e-3):
    # YOUR CODE HERE
    self.C = C
    self.power = power # degree of the polynomial kernel (d in the slides) 
    self.gamma = gamma # Kernel coefficient for "rbf" and "poly"
    self.coef = coef # coefficent of the polynomial kernel (r in the slides)
    self.eps = eps
    self.kernel_name = kernel_name  # implement for 'linear', 'poly' and 'rbf'
    self.kernel = None
    self.alpha_1 = None
    self.alpha_2 = None
    self.support_vectors = None
    self.support_vector_labels = None
    self.t = None

  def get_kernel(self, kernel_name):
    def linear(x1, x2): return x1 @ x2
    def polynomial(x1, x2): return (self.gamma * (x1 @ x2) + self.coef) ** self.power
    def rbf(x1, x2): return np.exp(-self.gamma*np.linalg.norm(x1-x2)**2)
    
    kernel_functions = {'linear': linear,
                        'poly': polynomial,
                        'rbf': rbf}

    return kernel_functions[kernel_name]
  
  def fit(self, X, y):
    # YOUR CODE HERE
    X = np.array(X)
    y = np.array(y)
    nr_samples, nr_features = np.shape(X)

    if not self.gamma:
      self.gamma = 1 / nr_features
    
    kernel_matrix = np.zeros((nr_samples, nr_samples))
    self.kernel = self.get_kernel(self.kernel_name)
    for i in range(nr_samples):
        for j in range(nr_samples):
          kernel_matrix[i, j] = self.kernel(X[i], X[j])
        
    n = nr_samples # for simplicity
    a1 = cvxpy.Variable(n)
    a2 = cvxpy.Variable(n)

    P = kernel_matrix
    objective = cvxpy.Minimize(
      (1/2)*cvxpy.quad_form(a1-a2, P) + self.eps * np.ones(n) @ (a1+a2) - y.T @ (a1-a2)
    )
    constraints = [np.zeros(n) <= a1, a1 <= np.ones(n)*self.C,
                  np.zeros(n) <= a2, a2 <= np.ones(n)*self.C,
                  np.ones(n) @ (a1-a2) == 0]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve()
    
    #get indexes of non-zero values
    alpha_1 = a1.value
    alpha_2 = a2.value
    d_alpha = alpha_1 - alpha_2
    idx = d_alpha > 1e-7
    self.d_alpha = d_alpha[idx]
    
    # get the support vectors
    self.support_vectors = X[idx]
    
    # get the corresponding labels
    self.support_vector_labels = y[idx]

    self.t = self.support_vector_labels[-1] \
     - sum([self.d_alpha[i] * self.kernel(self.support_vectors[i], self.support_vectors[-1])
            for i in range(len(self.d_alpha))])
  
  def predict(self, X):
    # YOUR CODE HERE
    X = np.array(X)
    y_pred = []
    for x in X:
      y = sum([self.d_alpha[i]*self.kernel(self.support_vectors[i], x)
          for i in range(len(self.d_alpha))]) + self.t
      y_pred.append(y)
    return y_pred
    


class LogisticRegression:
  def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64,
   threshold=0.5):
    # YOUR CODE HERE
    self.learning_rate = learning_rate # learning rate for the GD 
    self.nr_iterations = nr_iterations # number of iterations for GD
    self.batch_size = batch_size  # batch size for the GD
    self.threshold = threshold
    self.bar = progressbar.ProgressBar(widgets=widgets)
    self.W = None
  
  @staticmethod
  def sigmoid(x):
    return 1 / (1 + np.exp(x))

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)

    X = np.insert(X, 0, 1, axis=1)
    nr_samples, nr_features = X.shape

    # intitialize a random weights
    if self.W is None:
        self.W = 0.0001 * np.random.randn(nr_features)

    self.loss = []
    for i in self.bar(range(self.nr_iterations)):
      # select samples from the data according to the batch size
      # Hint: you can use np.random.choice to select indices 
      indx = np.random.choice(nr_samples, self.batch_size) # YOUR CODE HERE
      X_batch = X[indx, :] # n x m matrix, nr_sample := n, nr_features := m
      y_batch = y[indx]
      
      p = self.sigmoid(X_batch @ self.W)
      loss = - np.log(p) @ y_batch - np.log(1-p) @ (1-y_batch)
      gradient = np.mean(X_batch * (y_batch - p)[:, np.newaxis], axis=0)

      self.W -= self.learning_rate * gradient
      self.loss.append(loss) 
  
  def predict(self, X):
    # YOUR CODE HERE
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)
    return 1 * (self.sigmoid(X @ self.W) > self.threshold)
 


class SoftmaxClassifier:

  def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64):
    self.learning_rate = learning_rate # learning rate for the GD 
    self.nr_iterations = nr_iterations # number of iterations for GD
    self.batch_size = batch_size  # batch size for the GD
    self.bar = progressbar.ProgressBar(widgets=widgets)
    self.W = None  # weight matrix
  
  @staticmethod
  def softmax(z):
    # write the softmax function as it is writen in the notebook
    # YOUR CODE HERE
    if z.ndim==1:
      z = z[np.newaxis, :]
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
  
  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)

    # insert 1s as the first feature
    X = np.insert(X, 0, 1, axis=1)

    nr_samples, nr_features = X.shape  
    nr_classes = len(np.unique(y)) 

    # transform y into a one-hot encoded matrix and denote it with Y
    Y = pd.get_dummies(pd.Series(y)).values # YOUR CODE HERE

    # intitialize a random matrix of size (n x m) for the weights
    if self.W is None:
        self.W = 0.0001 * np.random.randn(nr_features, nr_classes)

    self.loss = []
    for i in self.bar(range(self.nr_iterations)):
      # select samples from the data according to the batch size
      # Hint: you can use np.random.choice to select indices 
      indx = np.random.choice(nr_samples, self.batch_size) # YOUR CODE HERE
      X_batch = X[indx, :] # n x m matrix, nr_sample := n, nr_features := m
      Y_batch = Y[indx, :] # n x k matrix, nr_classes := k
      
      # get the probability matrix (p) using X_batch and the current W matrix
      # Hint: you need to apply softmax function to get probabilities
      # it will be a matrix of size n x k
      p = self.softmax(X_batch @ self.W) # YOUR CODE HERE

      # get the loss (matrix) using the log(p) and Y_batch
      # think about the loss function formula (L) as a dot product
      # it will be a matrix of size n x n, 
      # where the diagonals are the losses per sample point
      loss_matrix = - np.log(p) @ Y_batch.T # YOUR CODE HERE
      
      # get the average loss across the batch
      loss = np.mean(loss_matrix.diagonal())

      # compute the gradient using the last formula in the notebook
      # don't forget to normalize the gradient with the batch size
      # since we took the normalized cross-entropy (mean instead of sum)
      # think about the gradient as a dot product
      # the result should be an m x k matrix (the same size as W)
      gradient = X_batch.T @ (p - Y_batch) / self.batch_size # YOUR CODE HERE     

      self.W -= self.learning_rate * gradient
      self.loss.append(loss)

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)

    # use the weight matrix W to obtain the probabilities with softmax
    prob = X @ self.W # YOUR CODE HERE n x k matrix
    # get the index of the highest probability per row as the prediction
    # you may want to use np.argmax here
    return np.argmax(prob, axis=1)# YOUR CODE HERE


