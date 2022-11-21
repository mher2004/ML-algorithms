class MyNaiveBayes:
    def __init__(self, smoothing=False):
        self.smoothing = smoothing

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        self.labels = self.y_train.unique()
        self.categories = self.X_train.columns
        self.priors = self.calculate_priors()
        self.likelihoods = self.calculate_likelihoods()

    def predict(self, X_test):
        prediction = []
        for i in range(X_test.shape[0]):
            predict = {}
            for j in self.labels:
                pred = 1
                for k in self.categories:
                    pred *= self.likelihoods[j][k][X_test.iloc[i][k]]
                predict[j] = pred * self.priors[j]
            prediction.append(max(predict, key=predict.get))
        return prediction
        

    def calculate_priors(self): 
        priors = {}
        if self.smoothing:
            for i in self.labels:
                priors[i] = (self.y_train.value_counts()[i]+1)/(len(self.y_train)+2)
        else:
            for i in self.labels:
                priors[i] = (self.y_train.value_counts()[i])/(len(self.y_train))
       
        return priors

    def calculate_likelihoods(self):

        likelihoods = {}
        for i in self.labels:
            label_data = self.X_train[self.y_train == i]
            likelihoods[i] = {}
            for j in self.categories:
                likelihoods[i][j] = {}
                for k in self.X_train[j].unique():
                    if self.smoothing:
                        likelihoods[i][j][k] = (label_data[j].value_counts()[k]+1)/(self.y_train.value_counts()[i]+2)

                    else:
                        likelihoods[i][j][k] = label_data[j].value_counts()[k]/self.y_train.value_counts()[i]
        return likelihoods