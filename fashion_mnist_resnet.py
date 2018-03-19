# importing required libraries
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, Input, Activation, Add
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

# resnet using the functional api

# start with a convolutional layer
inputs = Input(shape=INPUT_SHAPE)
output = Conv2D(32, kernel_size=(3, 3), padding="same") (inputs)

# add first residual block with two convolutional layers
res = output
res = BatchNormalization()(res)
res = Activation("relu")(res)
res = Conv2D(32, kernel_size=(3, 3), padding="same") (res)
res = BatchNormalization()(res)
res = Activation("relu")(res)
res = Conv2D(32, kernel_size=(3, 3), padding="same") (res)
output = Add()([res, output])
 
# add max pooling to reduce dimension
output = MaxPooling2D(pool_size=(2, 2)) (output)

# add second residual block with two convolutional layers
res = output
res = BatchNormalization()(res)
res = Activation("relu")(res)
res = Conv2D(32, kernel_size=(3, 3), padding="same") (res)
res = BatchNormalization()(res)
res = Activation("relu")(res)
res = Conv2D(32, kernel_size=(3, 3), padding="same") (res)
output = Add()([res, output])

# add max pooling to reduce dimension
output = MaxPooling2D(pool_size=(2, 2)) (output)

# add third residual block with two convolutional layers
res = output
res = BatchNormalization()(res)
res = Activation("relu")(res)
res = Conv2D(32, kernel_size=(3, 3), padding="same") (res)
res = BatchNormalization()(res)
res = Activation("relu")(res)
res = Conv2D(32, kernel_size=(3, 3), padding="same") (res)
output = Add()([res, output])

# add two tense layers to the end
output = BatchNormalization()(output)
output = Activation("relu")(output)
output = Dropout(0.5) (output)
output = Flatten() (output)
output = Dense(128, activation='relu') (output)
output = Dense(10, activation='softmax') (output)
model = Model(inputs, output)


model.summary()

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=Adam(lr=0.001),
             metrics=['accuracy'])

history1 = model.fit(X_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=10,
                     verbose=1,
                     validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ('Test loss:', 0.25492791481614113)
# ('Test accuracy:', 0.9125)

# Reduce the learning rate and continue training
K.set_value(model.optimizer.lr, 0.0001)

history2 = model.fit(X_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=10,
                     verbose=1,
                     validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ('Test loss:', 0.21316197064816952)
# ('Test accuracy:', 0.9315)

# Reduce the learning rate and continue training
K.set_value(model.optimizer.lr, 0.00001)

history3 = model.fit(X_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=10,
                     verbose=1,
                     validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ('Test loss:', 0.2140544271439314)
# ('Test accuracy:', 0.9313)


#####################################################
# visualizing the learning curves
vis.vis_learning_curves((history1, history2, history3), "loss_cnn1.png")
