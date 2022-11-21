import pickle
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import SelectPercentile, f_classif


class ViolaJones:
    def __init__(self, T=10):
        """
          Args:
            T: The number of weak classifiers which should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, training, pos_num, neg_num):
        """
        Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
        Args:
            training: An array of tuples.
                The first element is the numpy array of shape (m, n) representing the image.
                The second element is  its classification (1 or 0)
            pos_num: the number of positive samples
            neg_num: the number of negative samples
        """
        weights = np.zeros(len(training))
        training_data = []
        print("Computing integral images")
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)

        print("Building features")
        features = self.build_features(training_data[0][0].shape)
        print("Applying features to training examples")
        X, y = self.apply_features(features, training_data)
        print("Selecting best features")
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        print("Selected %d potential features" % len(X))

        for t in range(self.T):
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)

            # Compute the weights and normalize them
            # YOUR CODE HERE

            # YOUR CODE HERE

            # Compute model alpha
            alpha = # YOUR CODE HERE
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (
                str(clf), len(accuracy) - sum(accuracy), alpha))

    @staticmethod
    def train_weak(X, y, features, weights):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
        Args:
            X: A numpy array of  shape (len(features), len(training_data)).
                Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data).
                The ith element is the classification of the ith training example
            features: an array of tuples. Each tuple's first element is an array of the
                rectangle regions which positively contribute to the feature.
                The second element is an array of rectangle regions negatively contributing to the feature
            weights: A numpy array of shape len(training_data). The ith element is
                the weight assigned to the ith training example
        Returns:
            An array of weak classifiers
        """
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            # Selecting the best feature, threshold and polarity (flip)
            # You had better use your code from AdaBoost
            # HINT: you must define best feature as best_feature = features[index]
            # YOUR CODE HERE

            # YOUR CODE HERE

            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        print("Trained all %d classifiers" % len(classifiers))
        return classifiers

    @staticmethod
    def build_features(image_shape):
        """
        Builds the possible features given an image shape
        Args:
            image_shape: a tuple of form (height, width)
        Returns:
            An array of tuples.
                Each tuple's first element is an array of the rectangle regions which positively
            contribute to the feature.
                The second element is an array of rectangle regions negatively contributing
            to the feature
        """
        height, width = image_shape
        features = []
        for w in range(1, width + 1):
            for h in range(1, height + 1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        # 2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i + w, j, w, h)
                        if i + 2 * w < width:  # Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j + h, w, h)
                        if j + 2 * h < height:  # Vertically Adjacent
                            features.append(([immediate], [bottom]))

                        right_2 = RectangleRegion(i + 2 * w, j, w, h)
                        # 3 rectangle features
                        if i + 3 * w < width:  # Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j + 2 * h, w, h)
                        if j + 3 * h < height:  # Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        # 4 rectangle features
                        bottom_right = RectangleRegion(i + w, j + h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    @staticmethod
    def select_best(classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
        Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples.
                The first element is the numpy array of shape (m, n) representing the integral image.
                The second element is its classification (1 or 0)
            Returns:
                A tuple containing the best classifier, its error, and an array of its accuracy
        """
        # YOUR CODE HERE

        # YOUR CODE HERE
        return best_clf, best_error, best_accuracy

    @staticmethod
    def apply_features(features, training_data):
        """
        Maps features onto the training dataset
        Args:
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions
                which positively contribute to the feature.
                The second element is an array of rectangle regions negatively contributing to the feature
            training_data: An array of tuples.
                The first element is the numpy array of shape (m, n) representing the integral image.
                The second element is its classification (1 or 0)
        Returns:
            X: A numpy array of shape (len(features), len(training_data)).
                Each row represents the value of a single feature for each training example
                HINT: value is the difference between positive and negative region values
            y: A numpy array of shape len(training_data).
                The ith element is the classification of the ith training example
        """
        # NOTE: this takes long time so you can use tqdm to see the process if it works
        # YOUR CODE HERE

        # YOUR CODE HERE
        return X, y

    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified
            0 otherwise
        """
        # NOTE: you should compare sum(alphas*predictions) with sum(alphas)/2
        # because we used 1 and 0 for classes
        # (in case of 1 and -1 we were comparing with 0)
        # YOUR CODE HERE

        # YOUR CODE HERE

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename + ".pkl", 'rb') as f:
            return pickle.load(f)


class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Args:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier (this named `flip` in your AdaBoost hw)
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        """
        Classifies an integral image based on a feature f and the classifiers threshold and polarity
          Args:
            x: A 2D numpy array of shape (m, n) representing the integral image
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        # YOUR CODE HERE

        # YOUR CODE HERE

    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s, %s" % (
            self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))


class RectangleRegion:
    def __init__(self, x, y, width, height):
        """
            x: x coordinate of the upper left corner of the rectangle
            y: y coordinate of the upper left corner of the rectangle
            width: width of the rectangle
            height: height of the rectangle
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        """
        Computes the value of the Rectangle Region given the integral image
        Args:
            integral image : numpy array, shape (m, n)
        Returns:
            value of the given region
        """
        return ii[self.y + self.height][self.x + self.width] + ii[self.y][self.x] - (
                    ii[self.y + self.height][self.x] + ii[self.y][self.x + self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)

    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)


def integral_image(image):
    """
    Computes the integral image representation of a picture.
        Args:
            image : an numpy array with shape (m, n)

        HINT:
            You can use this formulas:
            1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
            2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0

    """
    # YOUR CODE HERE

    # YOUR CODE HERE
    return ii
