import numpy as np

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / np.sum(counts)
    return -np.sum(p * np.log2(p))

def calculate_gini(y):
  _, counts = np.unique(y, return_counts=True)
  p = counts / np.sum(counts)
  return 1 - np.sum(p ** 2) 

class DecisionNode():
  def __init__(self, feature_id=None, threshold=None,
                value=None, true_branch=None, false_branch=None):
      self.feature_id = feature_id          
      self.threshold = threshold          
      self.value = value                  
      self.true_branch = true_branch      
      self.false_branch = false_branch    

class DTClassifier():
  def __init__(self, impurity='entropy', min_samples_split=2, min_impurity=1e-7,
                max_depth=float("inf")):
      # Minimum n of samples to justify split
      self.min_samples_split = min_samples_split
      # The minimum impurity to justify split
      self.min_impurity = min_impurity
      # The maximum depth to grow the tree to
      self.max_depth = max_depth
      # Function to calculate impurity 
      self.impurity = self.impurity_function(impurity)
      # # Function to determine prediction of y at leaf
      # self.leaf_value_calculation = None      
      self.root = None  # Root node in dec. tree
      # self.probability = probability

  def impurity_function(self, impurity_name):
    impurity_functions = {'gini': calculate_gini,
                          'entropy': calculate_entropy}
    return impurity_functions[impurity_name]

  def calculate_impurity_gain(self, y, y1, y2):
      size_right = len(y1) 
      size_left = len(y2)
      parent_impurity = self.impurity(y)
      imp_gain = parent_impurity - \
      (size_right * self.impurity(y1) + size_left * self.impurity(y2)) / (size_left+size_right)
      return imp_gain
      
  def divide_on_feature(self, X, y, feature_id, threshold):
    true_indices = X[:,feature_id] >= threshold

    X_1, y_1 = X[true_indices], y[true_indices]
    X_2, y_2 = X[~true_indices], y[~true_indices]

    return X_1, y_1, X_2, y_2

  def majority_vote(self, y):
    uniques, counts = np.unique(y, return_counts=True)
    return uniques[np.argmax(counts)]
  
  # def get_probabilities(self, y):
  #   uniques, counts = np.unique(y, return_counts=True)
  #   if (len(uniques) == 1) and (uniques == 1)[0]:
  #     return np.array([0, 1])
  #   elif (len(uniques) == 1) and (uniques == 0)[0]:
  #     return np.array([1, 0])
  #   return counts / sum(counts)    

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
            X1, y1, X2, y2 = self.divide_on_feature(X, y, feature_id, threshold)                        

            if len(X1) > 0 and len(X2) > 0:
                # Calculate impurity
                impurity_gain = self.calculate_impurity_gain(y, y1, y2)

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

    # if self.probability:
      # leaf_value = self.get_probabilities(y)
    # else:  
    leaf_value = self.majority_vote(y)

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
      y_pred = np.array([self.predict_value(instance) for instance in X])
      return y_pred
  
  # def predict_proba(self, X):
  #   """ Classify samples one by one and return the set of labels """
  #   X = np.array(X)
  #   y_pred = [self.predict_value(instance)[1] for instance in X]
  #   return y_pred

