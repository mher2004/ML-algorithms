import numpy as np

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
  


