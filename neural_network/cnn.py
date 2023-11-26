import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from data_processing.data_service import load_data, load_input
from keras.models import load_model, save_model

from data_processing.mfcc import generate_mfcc


gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {gpus}')


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    return model


def compile_and_fit(model, x_train, x_test, y_train, y_test):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model


if __name__ == '__main__':
    model = load_model('test_model.h5')
    x = load_input(filepath='../data/zero.wav')
    prediction = np.argmax(model.predict(x))
    print(prediction)
