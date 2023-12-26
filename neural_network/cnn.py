import os

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from data_processing.data_service import load_data, load_input_from_file
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
    model.fit(x_train, y_train, batch_size=1, epochs=20, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model


def predict_directory(directory, model_path):
    model = load_model(model_path)
    # nb_of_positives = os.listdir(f'{directory}/{0}')
    correct = 0
    incorrect = 0
    for i in range(2):
        dirpath = f'{directory}/{i}'
        for file in os.listdir(dirpath):
            x = load_input_from_file(filepath=f'{dirpath}/{file}')
            result = model.predict(x, verbose=0)[0]
            prediction = np.argmax(result)
            percent = round(result[prediction] * 100, 2)
            if prediction == 0 and percent < 95:
                prediction = 1
            is_correct = prediction == i
            if is_correct:
                correct += 1
            else:
                incorrect +=1
            print(f'{file}: {percent}% {is_correct}')
    print(f'correct: {correct}\nincorrect: {incorrect}'
          f'\naccuracy: {round(correct/(correct+incorrect)*100,2)}%')


def predict_stream(model_path):
    model = load_model(model_path)



if __name__ == '__main__':
    #
    # x_train, x_test, labels_train, labels_test = (
    #     load_data('../data/eryk_training', 0.001))
    # input_shape = (x_train.shape[1], x_train.shape[2], 1)
    # output_shape = labels_train.shape[1]
    # model = create_model(input_shape, output_shape)
    # model = compile_and_fit(model, x_train, x_test, labels_train, labels_test)
    # save_model(model, 'eryk2.h5')
    predict_directory(directory='../data/eryk_training', model_path='eryk2.h5')

