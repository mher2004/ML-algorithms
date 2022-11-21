import numpy as np
import cvxpy as cp


class MyLinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):
    self.regularization = regularization
    self.lam = lam
    self.learning_rate = learning_rate
    self.tol = tol
    self.weights = None
  
  def fit(self, X, y):
    
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)

    if self.regularization is None:
      self.weights = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    elif self.regularization == 'l2':
      I = np.identity(X.shape[0])
      self.weights = np.linalg.pinv(X.T @ X + self.lam * I) @ (X.T @ y)
    elif self.regularization == 'l1':
      self.weights = np.random.randn(X.shape[1])
      converged = False
      self.loss = []
      i = 0
      while (not converged):
        i += 1
        y_pred = X @ self.weights
        self.loss.append(np.mean((y-y_pred)**2))
        grad = -2 * X.T @ (y-y_pred) + self.lam * np.sign(self.weights)
        new_weights = self.weights - self.learning_rate * grad
        converged = np.linalg.norm(self.weights - new_weights) < self.tol
        self.weights = new_weights
      print(f'Converged in {i} steps')

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)
    return X @ self.weights 
  

class SVR:
  def __init__(self, epsilon=0.1, C=1, kernel_name='linear', power=2, gamma=None, coef=2):
    self.C = C
    self.power = power 
    self.gamma = gamma 
    self.coef = coef 
    self.kernel_name = kernel_name  
    self.epsilon = epsilon
    self.kernel = None
    self.alphas_minus = None
    self.support_vectors = None
    self.support_vector_labels = None
    self.t = None

  def get_kernel(self, kernel_name):
    def linear(x1, x2): return np.dot(x1, x2.T)
    def polynomial(x1, x2): return (np.dot(self.gamma * x1, x2.T) + self.coef) ** self.power
    def rbf(x1, x2): return np.exp(-self.gamma * (np.linalg.norm(x1 - x2) ** 2))

    kernel_functions = {'linear': linear,
                        'poly': polynomial,
                        'rbf': rbf}

    return kernel_functions[kernel_name]

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)

    nr_samples, nr_features = np.shape(X)

    # Setting a default value for gamma
    if not self.gamma:
      self.gamma = 1 / nr_features

    # Set the kernel function
    self.kernel = self.get_kernel(self.kernel_name)

    # Construct the kernel matrix
    kernel_matrix = np.zeros((nr_samples, nr_samples))
    for i in range(nr_samples):
      for j in range(nr_samples):
        kernel_matrix[i, j] = self.kernel(X[i], X[j])

    # Define the quadratic optimization problem
    Q = kernel_matrix
    e = np.ones(nr_samples)
    a = cp.Variable(nr_samples)
    a_ = cp.Variable(nr_samples)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(a - a_, Q) - self.epsilon * e.T @ (a + a_) + y.T @ (a - a_)),
                    [0 <= a,
                     a <= self.C,
                     0 <= a_,
                     a_ <= self.C,
                     e.T @ (a - a_) == 0])

    # Solve the quadratic optimization problem using cvxopt
    prob.solve()

    alphas = a.value
    alphas_ = a_.value
    alpha_dif = alphas - alphas_
    idx = alpha_dif > 1e-7
    ind = np.arange(len(alphas))[idx]

    self.alpha_dif = alpha_dif[idx]
    self.support_vectors = X[idx]
    self.support_vector_labels = y[idx]

    # Calculate intercept (t) with first support vector
    self.t = self.support_vector_labels[0] - self.epsilon
    for i in range(len(self.alpha_dif)):
      self.t -= self.alpha_dif[i] * kernel_matrix[ind[i], 0]

  def predict(self, X):
    y_pred = []
    for instance in np.array(X):
      prediction = 0
      # determine the label of the given instance by the support vectors
      for i in range(len(self.alpha_dif)):
        prediction += self.alpha_dif[i] * self.kernel(self.support_vectors[i], instance)
      prediction += self.t
      y_pred.append(prediction)
    return np.array(y_pred)




def calculate_entropy(y):
  _, counts = np.unique(y, return_counts=True)
  p = counts / np.sum(counts)
  return -np.sum(p * np.log2(p))

def calculate_gini(y):
  _, counts = np.unique(y, return_counts=True)
  p = counts / np.sum(counts)
  return 1 - np.sum(p ** 2)   
  
def impurity_function(impurity_name):
  impurity_functions = {'gini': calculate_gini,
                        'entropy': calculate_entropy}
  return impurity_functions[impurity_name]
  
def divide_on_feature(X, y, feature_id, threshold):
  true_indices = X[:, feature_id] >= threshold
  X_1, y_1 = X[true_indices], y[true_indices]
  X_2, y_2 = X[~true_indices], y[~true_indices]
  return X_1, y_1, X_2, y_2


class DecisionNode():
  def __init__(self, feature_id=None, threshold=None,
                value=None, true_branch=None, false_branch=None):                
    self.feature_id = feature_id          
    self.threshold = threshold          
    self.value = value                  
    self.true_branch = true_branch      
    self.false_branch = false_branch    

class DecisionTree:
  def __init__(self, impurity='entropy', min_samples_split=2,
   min_impurity=1e-7, max_depth=float("inf")):
    self.min_samples_split = min_samples_split
    self.min_impurity = min_impurity
    self.max_depth = max_depth
    self.impurity = impurity_function(impurity)
    self.impurity_gain = None
    self.get_leaf_value = None            
    self.root = None  # Root node in dec. tree      

  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    self.root = self.build_tree(X, y)

  def build_tree(self, X, y, current_depth=0):
    largest_impurity_gain = 0

    nr_samples, nr_features = np.shape(X)

    if nr_samples >= self.min_samples_split and current_depth <= self.max_depth:
      for feature_id in range(nr_features):                      
        unique_values = np.unique(X[:, feature_id])
        thresholds = (unique_values[1:] + unique_values[:-1]) / 2
        # Iterate through all thresholds values of feature column i and
        # calculate the impurity
        for threshold in thresholds:          
          # Divide X and y depending on if the feature value of X at index feature_i
          # meets the threshold
          X1, y1, X2, y2 = divide_on_feature(X, y, feature_id, threshold)                        

          if len(X1) > 0 and len(X2) > 0:
            # Calculate impurity
            impurity_gain = self.impurity_gain(y, y1, y2)

            # If this threshold resulted in a higher information gain than previously
            # recorded save the threshold value and the feature
            # index
            if impurity_gain > largest_impurity_gain:
              largest_impurity_gain = impurity_gain
              best_feature_id = feature_id
              best_threshold = threshold
              best_X1 = X1 # X of right subtree (true)
              best_y1 = y1 # y of right subtree (true)
              best_X2 = X2 # X of left subtree (true)
              best_y2 = y2 # y of left subtree (true)

    if largest_impurity_gain > self.min_impurity:
      true_branch = self.build_tree(best_X1,
                                    best_y1,
                                    current_depth + 1)
      
      false_branch = self.build_tree(best_X2,
                                      best_y2,
                                      current_depth + 1)
      return DecisionNode(feature_id=best_feature_id,
                          threshold=best_threshold,
                          true_branch=true_branch,
                          false_branch=false_branch)

    leaf_value = self.get_leaf_value(y)

    return DecisionNode(value=leaf_value)


  def predict_value(self, x, tree=None):

    if tree is None:
        tree = self.root

    # If we have a value (i.e we're at a leaf) => return value as the prediction
    if tree.value is not None:
        return tree.value

    # Choose the feature that we will test
    feature_value = x[tree.feature_id]

    # Determine if we will follow left or right branch
    branch = tree.false_branch
    if feature_value >= tree.threshold:
      branch = tree.true_branch
    
    return self.predict_value(x, branch)

  def predict(self, X):
      """ Classify samples one by one and return the set of labels """
      X = np.array(X)
      y_pred = [self.predict_value(instance) for instance in X]
      return y_pred

class RegressionTree(DecisionTree):
  def calculate_variance_reduction(self, y, y1, y2):
    var_tot = np.var(y)
    var_1 = np.var(y1)
    var_2 = np.var(y2)
    frac_1 = len(y1) / len(y)
    frac_2 = len(y2) / len(y)

    variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

    return variance_reduction

  def mean_of_y(self, y):
    return np.mean(y)
    # value = np.mean(y)
    # return value if len(value) > 1 else value[0]

  def fit(self, X, y):
    self.impurity_gain = self.calculate_variance_reduction
    self.get_leaf_value = self.mean_of_y
    super().fit(X, y)

class DTClassifier(DecisionTree):
  def calculate_impurity_gain(self, y, y1, y2):
    size_right = len(y1) 
    size_left = len(y2)
    parent_impurity = self.impurity(y)
    imp_gain = parent_impurity - \
    (size_right * self.impurity(y1) + size_left * self.impurity(y2)) / (size_left+size_right)
    return imp_gain

  def majority_vote(self, y):
    uniques, counts = np.unique(y, return_counts=True)
    return uniques[np.argmax(counts)]
  
  def fit(self, X, y):
    self.impurity_gain = self.calculate_impurity_gain
    self.get_leaf_value = self.majority_vote
    super().fit(X, y)

class LogisticRegression:

  def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64):
   # YOUR CODE HERE
    self.learning_rate = learning_rate
    self.nr_iterations = nr_iterations
    self.batch_size = batch_size
    self.W = None
    self.bar = progressbar.ProgressBar(widgets=widgets)
    self.sigmoid = lambda x: 1 / (1 + np.exp(-x)) 

  def fit(self, X, y):
    # YOUR CODE HERE
    X = np.array(X)
    y = np.array(y)
    X = np.insert(X, 0, 1, axis=1)
    nr_samples, nr_features = X.shape
    Y = np.zeros((nr_samples, 2))
    Y[np.arange(nr_samples), y] = 1
    if self.W is None:
        self.W = 0.0001 * np.random.randn(nr_features, 2)
    self.loss = []
    for i in self.bar(range(self.nr_iterations)):
      indx = np.random.choice(np.arange(nr_samples), self.batch_size)
      X_batch = X[indx, :]
      Y_batch = Y[indx, :]
      p = self.sigmoid(X_batch @ self.W)
      self.W -= self.learning_rate * X_batch.T @ (p - Y_batch) / self.batch_size
      self.loss.append(np.mean((-Y_batch @ np.log(p).T).diagonal()))

  def predict(self, X):
    # YOUR CODE HERE
    X = np.insert(np.array(X), 0, 1, axis=1)
    return np.argmax(self.sigmoid(X @ self.W), axis=1)

class SoftmaxClassifier:

  def __init__(self, learning_rate=1e-3, nr_iterations=10,
   batch_size=64):
    self.learning_rate = learning_rate # learning rate for the GD 
    self.nr_iterations = nr_iterations # number of iterations for GD
    self.batch_size = batch_size  # batch size for the GD
    self.bar = progressbar.ProgressBar(widgets=widgets)
    self.W = None  # weight matrix
  
  @staticmethod
  def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax
  
  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)

    # insert 1s as the first feature
    X = np.insert(X, 0, 1, axis=1)

    nr_samples, nr_features = X.shape  
    nr_classes = len(np.unique(y)) 

    # transform y into a one-hot encoded matrix and denote it with Y
    # Y = # YOUR CODE HERE
    Y = np.zeros((nr_samples, nr_classes))
    Y[np.arange(nr_samples), y] = 1

    # intitialize a random matrix of size (n x m) for the weights
    if self.W is None:
        self.W = 0.0001 * np.random.randn(nr_features, nr_classes)

    self.loss = []
    for i in self.bar(range(self.nr_iterations)):
      # select samples from the data according to the batch size
      # Hint: you can use np.random.choice to select indices 
      indx = np.random.choice(np.arange(nr_samples), self.batch_size)# YOUR CODE HERE
      X_batch = X[indx, :] # n x m matrix, nr_sample := n, nr_features := m
      Y_batch = Y[indx, :] # n x k matrix, nr_classes := k

      # get the probability matrix (p) using X_batch and the current W matrix
      # Hint: you need to apply softmax function to get probabilities
      # it will be a matrix of size n x k
      p = SoftmaxClassifier.softmax(X_batch @ self.W)# YOUR CODE HERE

      # get the loss (matrix) using the log(p) and Y_batch
      # think about the loss function formula (L) as a dot product
      # it will be a matrix of size n x n, 
      # where the diagonals are the losses per sample point
      loss_matrix = -Y_batch @ np.log(p).T# YOUR CODE HERE

      # get the average loss across the batch
      loss = np.mean(loss_matrix.diagonal())

      # compute the gradient using the last formula in the notebook
      # don't forget to normalize the gradient with the batch size
      # since we took the normalized cross-entropy (mean instead of sum)
      # think about the gradient as a dot product
      # the result should be an m x k matrix (the same size as W)
      gradient = X_batch.T @ (p - Y_batch) / self.batch_size# YOUR CODE HERE     

      self.W -= self.learning_rate * gradient
      self.loss.append(loss)

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)

    # use the weight matrix W to obtain the probabilities with softmax
    prob = SoftmaxClassifier.softmax(X @ self.W)# YOUR CODE HERE n x k matrix
    # get the index of the highest probability per row as the prediction
    # you may want to use np.argmax here
    return np.argmax(prob, axis=1)# YOUR CODE HERE
