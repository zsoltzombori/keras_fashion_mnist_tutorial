# importing required libraries
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np

import vis

# set random seed
np.random.seed(12345)

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

# add extra channel dimension
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)
X_val = np.expand_dims(X_val, 3)

print("Train data shape ", X_train.shape)
print("Train label shape ", y_train.shape)
print("Validation data shape ", X_val.shape)
print("Validation label shape ", y_val.shape)
print("Test data shape ", X_test.shape)
print("Test label shape ", y_test.shape)

INPUT_SHAPE=X_train.shape[1:]
BATCH_SIZE = 512

# CNN with 1 Convolutional Layer
SEQUENTIAL=False
if SEQUENTIAL:
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
else: # alternative creation of the model using the functional api
    inputs = Input(shape=INPUT_SHAPE)
    output = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
    output = MaxPooling2D(pool_size=(2, 2)) (output)
    output = Dropout(0.2) (output)
    output = Flatten() (output)
    output = Dense(128, activation='relu') (output)
    output = Dense(10, activation='softmax') (output)
    model = Model(inputs, output)


model.summary()

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=Adam(lr=0.001),
             metrics=['accuracy'])

# Use data augmentation
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_gen = ImageDataGenerator()
val_batches = val_gen.flow(X_val, y_val, batch_size=BATCH_SIZE)

history1 = model.fit_generator(batches, steps_per_epoch=48000//BATCH_SIZE, epochs=20, 
                               validation_data=val_batches, validation_steps=12000//BATCH_SIZE)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Reduce the learning rate and continue training
K.set_value(model.optimizer.lr, 0.0001)

history2 = model.fit_generator(batches, steps_per_epoch=48000//BATCH_SIZE, epochs=20, 
                               validation_data=val_batches, validation_steps=12000//BATCH_SIZE)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Reduce the learning rate and continue training
K.set_value(model.optimizer.lr, 0.0001)

history2 = model.fit_generator(batches, steps_per_epoch=48000//BATCH_SIZE, epochs=20, 
                               validation_data=val_batches, validation_steps=12000//BATCH_SIZE)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#####################################################
# visualizing the learning curves
vis.vis_learning_curves((history1, history2, history3), "loss_cnn1_aug.png")
