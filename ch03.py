import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score, cross_val_predict


# print(datasets.get_data_home())

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1], "g-", label="재현율")
    plt.xlabel("임계값")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


mnist = fetch_mldata('MNIST original')
print(mnist)

X, y = mnist["data"], mnist["target"]
print(X.shape)
print(y.shape)

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
#
# print(y[36000])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

suffle_index = np.random.permutation(60000)
X_train, y_train = X_train[suffle_index], y_train[suffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(cvs)

never_5_clf = Never5Classifier()
cvs = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(cvs)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(confusion_matrix(y_train_5, y_train_pred))

# print(precision_score(y_train_5, y_train_pred))
# print(recall_score(y_train_5, y_train_pred))
# print(f1_score(y_train_5, y_train_pred))

y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print(precisions, recalls, thresholds)
