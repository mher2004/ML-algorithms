import numpy as np
import progressbar

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

class RegressionTree:
   def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
   # YOUR CODE HERE
   
   def fit(self, X, y):
   # YOUR CODE HERE
   
   def predict(self, X):
   # YOUR CODE HERE


class SVR:
   def __init__(self, epsilon=0.1, C=1, kernel_name='linear', power=2, gamma=None, coef=2):
   # YOUR CODE HERE
   
   def fit(self, X, y):
   # YOUR CODE HERE
   
   def predict(self, X):
   # YOUR CODE HERE


class LogisticRegression:
   def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64):
   # YOUR CODE HERE
   
   def fit(self, X, y):
   # YOUR CODE HERE
   
   def predict(self, X):
   # YOUR CODE HERE
 


class SoftmaxClassifier:

  def __init__(self, learning_rate=1e-3, nr_iterations=10,
   batch_size=64):
    self.learning_rate = learning_rate # learning rate for the GD 
    self.nr_iterations = nr_iterations # number of iterations for GD
    self.batch_size = batch_size  # batch size for the GD
    self.bar = progressbar.ProgressBar(widgets=widgets)
    self.W = None  # weight matrix
  
  @staticmethod
  def softmax(z):
  # write the softmax function as it is writen in the notebook
  # YOUR CODE HERE
  
  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)

    # insert 1s as the first feature
    X = np.insert(X, 0, 1, axis=1)

    nr_samples, nr_features = X.shape  
    nr_classes = len(np.unique(y)) 

    # transform y into a one-hot encoded matrix and denote it with Y
    # Y = # YOUR CODE HERE

    # intitialize a random matrix of size (n x m) for the weights
    if self.W is None:
        self.W = 0.0001 * np.random.randn(nr_features, nr_classes)

    self.loss = []
    for i in self.bar(range(self.nr_iterations)):
      # select samples from the data according to the batch size
      # Hint: you can use np.random.choice to select indices 
      indx = # YOUR CODE HERE
      X_batch = X[indx, :] # n x m matrix, nr_sample := n, nr_features := m
      Y_batch = Y[indx, :] # n x k matrix, nr_classes := k

      # get the probability matrix (p) using X_batch and the current W matrix
      # Hint: you need to apply softmax function to get probabilities
      # it will be a matrix of size n x k
      p = # YOUR CODE HERE

      # get the loss (matrix) using the log(p) and Y_batch
      # think about the loss function formula (L) as a dot product
      # it will be a matrix of size n x n, 
      # where the diagonals are the losses per sample point
      loss_matrix = # YOUR CODE HERE

      # get the average loss across the batch
      loss = np.mean(loss_matrix.diagonal())

      # compute the gradient using the last formula in the notebook
      # don't forget to normalize the gradient with the batch size
      # since we took the normalized cross-entropy (mean instead of sum)
      # think about the gradient as a dot product
      # the result should be an m x k matrix (the same size as W)
      gradient = # YOUR CODE HERE     

      self.W -= self.learning_rate * gradient
      self.loss.append(loss)

  def predict(self, X):
    X = np.array(X)
    X = np.insert(X, 0, 1, axis=1)

    # use the weight matrix W to obtain the probabilities with softmax
    prob = # YOUR CODE HERE n x k matrix
    # get the index of the highest probability per row as the prediction
    # you may want to use np.argmax here
    return # YOUR CODE HERE


