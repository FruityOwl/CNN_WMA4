import os
import cv2 as cv
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    images = []
    labels = []
    descriptors = []

    for folder in os.listdir("db"):
        for file in os.listdir(os.path.join("db", folder)):
            img = os.path.join(os.path.join("db", folder), file)
            img = cv.imread(img)
            img = cv.resize(img, (150, 150))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            images.append(np.array(img, dtype='float32'))
            descriptor = cv.HuMoments(cv.moments(gray)).flatten()
            descriptors.append(descriptor)

            if folder == "Banana":
                labels.append(np.array(0, dtype='int32'))
            elif folder == "Lemon":
                labels.append(np.array(1, dtype='int32'))
            else:
                labels.append(np.array(2, dtype='int32'))
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(150, 150, 3), name='Conv2D-1'),
        layers.MaxPooling2D(pool_size=2, name='MaxPool'),
        layers.Dropout(0.2, name='Dropout'),
        layers.Flatten(name='flatten'),
        layers.Dense(32, activation='relu', name='Dense'),
        layers.Dense(3, activation='softmax', name='Output')
    ], name='First_Layer')

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32)
    print("test: ")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(test_loss)
    print(descriptors)


if __name__ == '__main__':
    main()
