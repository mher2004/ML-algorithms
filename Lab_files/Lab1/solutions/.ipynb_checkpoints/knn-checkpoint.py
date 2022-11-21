import numpy as np

class KNearestNeighbor(object):

  """ a kNN classifier with arbitrary distance """

  def __init__(self, X_train, y_train):
    """
    Initializing the KNN object

    Inputs:
    - X_train: A numpy array or pandas DataFrame of shape (num_train, D) 
    - y_train: A numpy array or pandas Series of shape (N,) containing the training labels
    """
    self.X_train = X_train
    self.y_train = y_train
    
  def fit_predict(self, X_test, k=1):
    """
    This method fits the data and predicts the labels for the given test data.
    For k-nearest neighbors fitting (training) is just 
    memorizing the training data. 
    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) 
    - k: The number of nearest neighbors.
    - num_loops: number of for loop implementation.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing predicted labels
    """
    dists = self.compute_distances(X_test)
    return self.predict_labels(dists, k=k)

  def compute_distances(self, X_test):
    """
    Compute the distance between each test point in X_test and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    ##### YOUR CODE STARTS HERE ##### 
    X_train = self.X_train
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        dists[i, :] = (X_train != X_test.iloc[i,:]).sum(axis=1)
    ##### YOUR CODE ENDS HERE #####          
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance between the ith test point and the jth training point.

    Returns:
    - y: A numpy array or pandas Series of shape (num_test,) containing the
    predicted labels for the test data  
    """
    ##### YOUR CODE STARTS HERE ##### 
    def majority_class(y): return y.value_counts().index[0]

    if k == 1:
      y_pred = self.y_train.iloc[dists.argmin(axis=1)]
    else:
      y_pred = []
      for i in range(dists.shape[0]):
        closest_k = np.argsort(dists[i, :])[:k]
        closest_y = self.y_train.iloc[closest_k]
        y_pred.append(majority_class(closest_y))
    ##### YOUR CODE ENDS HERE #####              
    return np.array(y_pred)