import pickle
import time
from viola_jones import ViolaJones


def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1

        correct += 1 if prediction == y else 0

    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives / all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives / all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct / len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))


if __name__ == "__main__":
    filename = "viola_jones"
    training = True
    small = True  # set True to use small dataset for experiments
    train_path = small * "small_" + "training.pkl"
    test_path = small * "small_" + "test.pkl"
    with open(train_path, 'rb') as f_train, open(test_path, 'rb') as f_test:
        train = pickle.load(f_train)
        test = pickle.load(f_test)
    print(f"Train data has {len(train)} instances.")
    print(f"Test data has {len(test)} instances.")

    # Calculate number of pos and neg samples
    targets = list(zip(*train)[1])
    pos_num = sum(targets)
    neg_num = len(targets) - pos_num
    print(f"There is {pos_num} positive and {neg_num} negative samples.")

    # Initialize viola jones model object
    vj = ViolaJones(T=10)

    if training:
        # Train vj model on training dataset
        vj.train(training=train, pos_num=pos_num, neg_num=neg_num)

        # Save trained vj model
        vj.save(filename=filename)
    else:
        # Load already saved model
        vj = ViolaJones.load(filename=filename)

    # Test vj model on test dataset
    evaluate(clf=vj, data=test)
