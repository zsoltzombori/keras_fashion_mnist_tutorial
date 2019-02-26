# importing required libraries
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import numpy as np

import vis

# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# convert brightness values from bytes to floats between 0 and 1:
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# save last 20% from x_train, y_train for validation set (for model tuning)
val_size = len(X_train) // 5
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

print("Train data shape ", X_train.shape)
print("Train label shape ", y_train.shape)
print("Validation data shape ", X_val.shape)
print("Validation label shape ", y_val.shape)
print("Test data shape ", X_test.shape)
print("Test label shape ", y_test.shape)

INPUT_SHAPE=X_train.shape[1:]
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NB_EPOCHS = 30

# Two layer dense network
model = Sequential([
    Flatten(input_shape=INPUT_SHAPE),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(lr=LEARNING_RATE),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


history = model.fit(X_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=NB_EPOCHS,
                     verbose=1,
                     validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ('Test loss:', 0.33005201976299287)
# ('Test accuracy:', 0.8925)

#####################################################
# visualizing the learning curves
vis.vis_learning_curves((history,), "loss_dense.png")


#####################################################
# Visualizing the classification:
y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred, axis=1)

vis.vis_classification(X_test, y_pred, y_test, bucket_size=10, nb_classes=10, file_name="results_dense.png")
