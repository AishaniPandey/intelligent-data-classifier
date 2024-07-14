import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class KNN:
    def __init__(self, k=3, encoder_type="resnet", distance_metric="euclidean"):
        self.k = k
        self.encoder_type = encoder_type
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.label_to_int = {}
        self.int_to_label = {}

    def fit(self, X, y):
        unique_labels = np.unique(y)
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}
        y_int = np.array([self.label_to_int[label] for label in y])
        self.X_train = X
        self.y_train = y_int

    def predict(self, X):
        y_pred_int = [self._predict(x) for x in X]
        y_pred = [self.int_to_label[y_int] for y_int in y_pred_int]
        return np.array(y_pred)

    def _predict(self, x):
        if self.distance_metric == "euclidean":
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        elif self.distance_metric == "manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        elif self.distance_metric == "cosine":
            dot_product = np.dot(self.X_train, x)
            norm_x = np.linalg.norm(x)
            norm_X_train = np.linalg.norm(self.X_train, axis=1)
            distances = 1 - (dot_product / (norm_x * norm_X_train))
        elif self.distance_metric == "hamming":
            distances = np.sum(self.X_train != x, axis=1)
        elif self.distance_metric == "minkowski":
            p = 3  # You can make 'p' a parameter if you like
            distances = np.sum(np.abs(self.X_train - x)**p, axis=1)**(1/p)
        else:
            raise ValueError("Invalid distance metric.")

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
        # # ... (your distance metric code)

        # k_indices = np.argsort(distances)[:self.k]
        # k_nearest_labels = [self.y_train[i] for i in k_indices]

        # most_common = np.bincount(k_nearest_labels).argmax()
        # return most_common

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        #MIght need to comment this out.
        y_pred = y_pred.astype(str)

        f1 = f1_score(y, y_pred, average='macro', zero_division=1)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro', zero_division=1)
        recall = recall_score(y, y_pred, average='macro', zero_division=1)

        return {"f1_score": f1, "accuracy": accuracy, "precision": precision, "recall": recall}

def measure_time(model, X_test):
    start_time = time.time()
    predictions = model.predict(X_test)
    return time.time() - start_time
