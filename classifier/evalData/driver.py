import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from KnnClassToRun import KNN, measure_time
import sys
import time

if __name__ == "__main__":

    # Load training data
    X_resnet_train = np.load('X_resnet_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)

    # Load test data from command line argument
    if len(sys.argv) <= 1:
        print("Usage: python driver.py <path_to_test_dataset>")
        exit(1)

    data = np.load(sys.argv[1], allow_pickle=True)
    X_resnet = data[:, 1]
    # X_vit = data[:, 2]
    y_test = data[:, 3]
    X_resnet_2D = np.vstack(X_resnet).reshape(-1, 1024)
    # X_vit_2D = np.vstack(X_vit).reshape(-1, 512)
    X_resnet_test = X_resnet_2D

    # #Debugging
    # X_resnet_test = np.load(sys.argv[1], allow_pickle=True)
    # y_test = np.load(sys.argv[2], allow_pickle=True)

    # Initialize models
    # initial_knn = KNN(k=3)
    best_knn = KNN(k=1,encoder_type='resnet',distance_metric='manhattan')  
    # optimized_knn = KNN(k=10)  
    # sklearn_knn = KNeighborsClassifier(n_neighbors=10)

    # Time measurement function
    def measure_time(model, X_test):
        start_time = time.time()
        predictions = model.predict(X_test)
        return time.time() - start_time

    # Initialize variables for plotting
    train_sizes = np.linspace(0.1, 1.0, 10)
    # times_initial = []
    times_best = []
    # times_optimized = []
    # times_sklearn = []

    # Measure inference time vs training data size
    for size in train_sizes:
        if size == 1.0:
            X_train, y_train_subset = X_resnet_train, y_train
        else:
            X_train, _, y_train_subset, _ = train_test_split(X_resnet_train, y_train, train_size=size, random_state=42)

        # initial_knn.fit(X_train, y_train_subset)
        best_knn.fit(X_train, y_train_subset)
        # optimized_knn.fit(X_train, y_train_subset)
        # sklearn_knn.fit(X_train, y_train_subset)

        # times_initial.append(measure_time(initial_knn, X_resnet_test))
        times_best.append(measure_time(best_knn, X_resnet_test))
        # times_optimized.append(measure_time(optimized_knn, X_resnet_test))
        # times_sklearn.append(measure_time(sklearn_knn, X_resnet_test))

    # # Plotting inference time vs training data size
    # plt.figure(figsize=(10, 6))
    # # plt.plot(train_sizes, times_initial, label='Initial KNN')
    # plt.plot(train_sizes, times_best, label='Best KNN')
    # # plt.plot(train_sizes, times_optimized, label='Optimized KNN')
    # # plt.plot(train_sizes, times_sklearn, label='Scikit-learn KNN')
    # plt.xlabel('Training Data Size')
    # plt.ylabel('Time (seconds)')
    # plt.title('Inference Time vs Training Data Size')
    # plt.legend()
    # plt.show()

    # Measure inference time for each model
    # initial_knn.fit(X_resnet_train, y_train)
    best_knn.fit(X_resnet_train, y_train)
    # optimized_knn.fit(X_resnet_train, y_train)
    # sklearn_knn.fit(X_resnet_train, y_train)

    # time_initial = measure_time(initial_knn, X_resnet_test)
    time_best = measure_time(best_knn, X_resnet_test)
    # time_optimized = measure_time(optimized_knn, X_resnet_test)
    # time_sklearn = measure_time(sklearn_knn, X_resnet_test)

    results=[]

    y_pred = best_knn.predict(X_resnet_test)

    y_pred = y_pred.astype(str)

    evaluation = best_knn.evaluate( X_resnet_test, y_test)
    accuracy = evaluation['accuracy']
    scoreee= evaluation["f1_score"]
    results.append((scoreee, accuracy, 1, "resnet", "manhattan"))

    for idx, (f1_score, accuracy, k, encoder, distance) in enumerate(results[:]):
        print(f"{idx + 1}. f1_score: {f1_score:.4f}, Accuracy: {accuracy:.4f}, k: {k}, Encoder: {encoder}, Distance: {distance}")


    # # Plotting inference time for initial KNN model, best KNN model, most optimized KNN model, and the default sklearn KNN model.
    # labels = [ 'Best KNN']
    # times = [ time_best]

    # plt.figure(figsize=(10, 6))


    # plt.figure(figsize=(10, 6))
    # plt.barh(labels, times, color='dodgerblue')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Model')
    # plt.title('Inference Time for Different KNN Models')
    # plt.gca().invert_yaxis()  # This is to have the Initial KNN at the top and Scikit-learn KNN at the bottom
    # plt.show()