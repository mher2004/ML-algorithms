import numpy as np


class MyNaiveBayes:
  def __init__(self, smoothing=False):
      self.smoothing = smoothing
    
  def fit(self, X_train, y_train):
      # use this method to learn the model
      # if you feel it is easier to calculate priors 
      # and likelihoods at the same time
      # then feel free to change this method
      self.X_train = X_train
      self.y_train = y_train
      self.priors = self.calculate_priors()
      self.likelihoods = self.calculate_likelihoods()      
      
  def predict(self, X_test):
      # recall: posterior is P(label_i|feature_j)
      # hint: Posterior probability is a matrix of size 
      #       m*n (m samples and n labels)
      #       our prediction for each instance in data is the class that 
      #       has the highest posterior probability. 
      #       You do not need to normalize your posterior, 
      #       meaning that for classification, prior and likelihood are enough
      #       and there is no need to divide by evidence. Think why!
      # return: a list of class labels (predicted)
      ##### YOUR CODE STARTS HERE ##### 
      likelihoods = self.likelihoods
      priors = self.priors
      labels = self.y_train.unique()
      nr_test = X_test.shape[0]
      prediction = []
      for i in range(nr_test):
        instance = X_test.iloc[i,:]
        probs = []
        for label in labels:
          prob = priors[priors.index == label]
          for idx, feature in enumerate(instance):
            prob *= likelihoods[f'{idx}={feature}|{label}']
          probs.append(prob)
        prediction.append(labels[np.argmax(probs)])
      ##### YOUR CODE ENDS HERE #####       
      return np.array(prediction)

  def calculate_priors(self):
      # recall: prior is P(label=l_i)
      # hint: store priors in a pandas Series or a list
      ##### YOUR CODE STARTS HERE #####             
      priors = (self.y_train.value_counts())/(len(self.y_train))
      ##### YOUR CODE ENDS HERE #####         
      return priors
  
  def calculate_likelihoods(self):
      # recall: likelihood is P(feature=f_j|label=l_i)
      # hint: store likelihoods in a data structure like dictionary:
      #        feature_j = [likelihood_k]
      #        likelihoods = {label_i: [feature_j]}
      #       Where j implies iteration over features, and 
      #             k implies iteration over different values of feature j. 
      #       Also, i implies iteration over different values of label. 
      #       Likelihoods, is then a dictionary that maps different label 
      #       values to its corresponding likelihoods with respect to feature
      #       values (list of lists).
      #
      #       NB: The above pseudocode is for the purpose of understanding
      #           the logic, but it could also be implemented as it is.
      #           You are free to use any other data structure 
      #           or way that is convenient to you!
      #
      #       More Coding Hints: You are encouraged to use Pandas as much as
      #       possible for all these parts as it comes with flexible and
      #       convenient indexing features which makes the task easier.
      ##### YOUR CODE STARTS HERE ##### 
      X_train = self.X_train
      y_train = self.y_train
      labels = y_train.unique()
      nr_labels = len(labels)
      smoothing = self.smoothing
      nr_features = X_train.shape[1]
      likelihoods = {}
      for label in labels:
        for col_id in range(nr_features):
          feature = X_train.iloc[:,col_id]
          levels = feature.unique()
          for level in levels:
            label_mask = y_train == label
            likelihoods[f'{col_id}={level}|{label}'] = ((
                (feature == level) & (label_mask)
                ).sum() + 1 * smoothing)/ (
                    (label_mask).sum() + len(levels) * smoothing)
      ##### YOUR CODE ENDS HERE ##### 
      return likelihoods