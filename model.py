import tensorflow as tf
from keras import datasets, layers, models 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05, 
    height_shift_range=0.05, 
    horizontal_flip=True,
    zoom_range=0.1
)
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
classes=["airplane", "automobile", "bird", "cat", "dear", "dog", "frog", "house", "ship", "truck"]
y_train = y_train.reshape(-1,)
#print(y_train[:5])
def plot_sample(X, y, index):
    plt.figure(figsize=(5,5))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()
#plot_sample(X_train, y_train, 5)
X_train=X_train/255
X_test=X_test/255

#ann = models.Sequential([
#    layers.Flatten(input_shape=(32,32,3)),
#    layers.Dense(3000, activation='relu'),
#    layers.Dense(1000, activation='relu'),
#    layers.Dense(10,activation='sigmoid')
#])

#ann.compile(optimizer='SGD',
 #           loss='sparse_categorical_crossentropy',
  #          metrics=['accuracy'])
#ann.fit(X_train, y_train, epochs=5)

#ann.evaluate(X_test, y_test)

cnn = models.Sequential([
    layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3), input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=128, activation='relu', kernel_size=(3,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(filters=256, activation='relu', kernel_size=(3,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
optimizer = Adam(learning_rate=0.0001)
cnn.compile(optimizer= optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen.fit(X_train)
cnn.fit(datagen.flow(X_train, y_train, batch_size=64), epochs=20, validation_data=(X_test, y_test))

cnn.evaluate(X_test, y_test)

# Predict on the test set
y_pred = cnn.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Visualize some predictions
for i in range(10):  # Show first 10 images
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[i])
    plt.title(f"True: {classes[y_test[i][0]]}, Predicted: {classes[y_pred_classes[i]]}")
    plt.axis('off')
    plt.show()

cnn.save('cifar10_model.h5')    