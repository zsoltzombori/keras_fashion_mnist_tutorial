# importing required libraries
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
val_size = len(X_train) / 5
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

# add extra channel dimension
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)
X_val = np.expand_dims(X_val, 3)

print "Train data shape ", X_train.shape
print "Train label shape ", y_train.shape
print "Validation data shape ", X_val.shape
print "Validation label shape ", y_val.shape
print "Test data shape ", X_test.shape
print "Test label shape ", y_test.shape

INPUT_SHAPE=X_train.shape[1:]
BATCH_SIZE = 512

# normalize data
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def norm_input(x): 
    return (x-mean_px)/std_px

# CNN with 4 Convolutional Layers and Batch Normalization
model = Sequential([
    Lambda(norm_input, input_shape=INPUT_SHAPE),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),    
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
             optimizer=Adam(lr=0.001),
             metrics=['accuracy'])

model.fit(X_train, y_train,
         batch_size=BATCH_SIZE,
         epochs=10,
         verbose=1,
         validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ('Test loss:', 0.23384492818117142)
# ('Test accuracy:', 0.922)

# Use data augmentation and continue training
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_batches = gen.flow(X_val, y_val, batch_size=BATCH_SIZE)

model.fit_generator(batches, steps_per_epoch=48000//BATCH_SIZE, epochs=50, 
                   validation_data=val_batches, validation_steps=12000//BATCH_SIZE)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ('Test loss:', 0.19834078181684017)
# ('Test accuracy:', 0.9311)
