import numpy as np
import cvxopt  # library for convex optimization

# hide cvxopt output
cvxopt.solvers.options['show_progress'] = False

class SVM:
  """
  Hard (C=None) and Soft (C>0) Margin Support Vector Machine classifier
  """
  def __init__(self, C=1):
      self.C = C
      self.alphas = None
      self.support_vectors = None
      self.support_vector_labels = None
      self.w = None
      self.t = None

  def fit(self, X, y):

      nr_samples, nr_features = np.shape(X)

      # Define the quadratic optimization problem
      P = cvxopt.matrix(np.outer(y, y) * (X @ X.T), tc='d')
      q = cvxopt.matrix(np.ones(nr_samples) * -1)
      A = cvxopt.matrix(y, (1, nr_samples), tc='d')
      b = cvxopt.matrix(0, tc='d')

      if not self.C:
          G = cvxopt.matrix(np.identity(nr_samples) * -1)
          h = cvxopt.matrix(np.zeros(nr_samples))
      else:
          G_max = np.identity(nr_samples) * -1
          G_min = np.identity(nr_samples)
          G = cvxopt.matrix(np.vstack((G_max, G_min)))
          h_max = cvxopt.matrix(np.zeros(nr_samples))
          h_min = cvxopt.matrix(np.ones(nr_samples) * self.C)
          h = cvxopt.matrix(np.vstack((h_max, h_min)))

      # Solve the quadratic optimization problem using cvxopt
      minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

      # Lagrange multipliers (denoted by alphas in the lecture slides)
      alphas = np.ravel(minimization['x'])

      # first get indexes of non-zero lagr. multipiers
      idx = alphas > 1e-7

      # get the corresponding lagr. multipliers (non-zero alphas)
      self.alphas = alphas[idx]

      # get the support vectors
      self.support_vectors = X[idx]
      
      # get the corresponding labels
      self.support_vector_labels = y[idx]

      self.w = self.alphas * self.support_vector_labels @ self.support_vectors

      # calculate the intercept (denoted by t in our lecture slides) with first support vector
      self.t = self.w @ self.support_vectors[0] - self.support_vector_labels[0]

  def predict(self, X):
      return np.sign(self.w @ X.T - self.t)