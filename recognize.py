import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle


def main():

    # Import and flatten data
    path = 'yalefaces'
    image_paths = [os.path.join(path, fname) for fname in os.listdir(path) if fname.startswith('subject')]
    image_paths = sorted(image_paths)
    image_matrix = np.array([
        np.array(Image.open(image_paths[i]).convert('L')).flatten()
        for i in range(len(image_paths)) ], 'f')
    # print(image_matrix.shape)

    # PCA
    eigen_values, eigen_vectors, mean_values = pca(image_matrix)

    # Plot eigen values
    plt.plot(eigen_values)
    plt.title('Eigen Values')
    plt.xlabel('Principal components')
    plt.ylabel('Lambda')
    plt.show()
    plt.clf()

    # Components needed to capture 50% energy
    total_energy = np.sum(eigen_values)
    energy = 0
    for k in range(len(eigen_values)):
        energy += eigen_values[k]
        print(energy/total_energy)
        if energy >= 0.5*total_energy:
            print(k+1, "components are needed to capture 50% energy")
            break

    # Plot Top 10 Eigen faces
    for k in range(10):
        plt.title('Eigen Face '+ str(k+1))
        plt.gray()
        plt.imshow(eigen_vectors.T[k].reshape(243,320))
        plt.show()
        plt.clf()

    # Image Reconstruction
    k_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for k in range(2):
        # Select a random image
        ori_image = image_matrix[random.randint(0, len(image_matrix)-1)]
        for k in k_values:
            c = np.dot(eigen_vectors.T[:k, :], ori_image)
            new_image = np.dot(eigen_vectors[:, :k], c)
            plt.title('k = ' + str(k))
            plt.gray()
            plt.imshow(new_image.reshape(243, 320))
            plt.show()
            plt.clf()

    # Face Recognition
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    output = np.array([re.findall(r'\d+', str(image_paths[i])) for i in range(len(image_paths))])

    for k in range(15):
        x_train, x_test, y_train, y_test = train_test_split(image_matrix[k*11: (k+1)*11],
                                                            output[k*11: (k+1)*11], test_size=0.25)
        X_train.extend(x_train)
        X_test.extend(x_test)
        Y_train.extend(y_train)
        Y_test.extend(y_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train).reshape(-1,).astype(int)
    Y_test = np.array(Y_test).reshape(-1,).astype(int)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Random shuffle train data
    X_train, Y_train = shuffle(X_train, Y_train)

    train_input = np.dot(X_train, eigen_vectors)
    test_input = np.dot(X_test, eigen_vectors)
    for k in k_values: # Different PCA components
        accuracy = knn(train_input[:, :k], test_input[:, :k], Y_train, Y_test)
        print("Face Recognition Accuracy on Test set is", accuracy, "for", k, "components")

    # CNN
    X_train = X_train.reshape(X_train.shape[0], 243, 320, 1)
    X_test = X_test.reshape(X_test.shape[0], 243, 320, 1)
    Y_train -= 1
    Y_test -= 1
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    model = create_model()
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=100, validation_split=0.2, shuffle=True)
    score, acc = model.evaluate(X_test, Y_test)
    print('CNN Test accuracy:', acc)

    # Data Augmentation
    print()
    print('Data Augmentation')
    datagen = ImageDataGenerator(
        # featurewise_center=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2)

    datagen.fit(X_train)
    new_model = create_model()
    new_model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train)/32, epochs=10)
    score, acc = new_model.evaluate(X_test, Y_test)
    print('CNN Test accuracy with Data Augmentation:', acc)


def pca(data_matrix):
    # Normalize
    mean_values = np.mean(data_matrix, axis=0)
    data_matrix -= mean_values
    stdev = np.std(data_matrix, axis=0)
    # stdev = np.array([i if i!=0 else 1 for i in stdev]) # Avoid / by 0
    # data_matrix /= stdev

    covariance = np.dot(data_matrix, data_matrix.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance)
    eigen_vectors = np.dot(data_matrix.T, eigen_vectors)
    for i in range(eigen_vectors.shape[1]):
        eigen_vectors[:, i] = eigen_vectors[:, i]/np.linalg.norm(eigen_vectors[:, i])
    return eigen_values.astype(float), eigen_vectors.astype(float), mean_values


def knn(x_train, x_test, y_train, y_test):
    best_n = 1
    best_acc = 0
    for n in range(1, 10):
        knn_clf = KNeighborsClassifier(n_neighbors=n)
        scores = cross_val_score(knn_clf, x_train, y_train, cv=5)
        acc = np.mean(scores)
        if acc > best_acc:
            best_acc = acc
            best_n = n
    # print(best_n)
    knn_clf = KNeighborsClassifier(n_neighbors=best_n)
    knn_clf.fit(x_train, y_train)
    accuracy = knn_clf.score(x_test, y_test)
    return accuracy

def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(243,320,1)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(5000, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(15, activation='softmax'))

    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

main()