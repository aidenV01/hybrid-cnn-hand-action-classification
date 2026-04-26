import numpy as np
import tensorflow as tf

assert tf.__version__.startswith('2')
from mediapipe_model_maker import gesture_recognizer

from tensorflow.keras.layers import (
    Conv2D,
    Input,
    Dropout,
    BatchNormalization,
    Flatten,
    Dense,
    Activation,
    MaxPooling2D,
    GlobalAveragePooling2D,
    LeakyReLU,
    Concatenate
)
from tensorflow.keras.models import Sequential

from sklearn.metrics import confusion_matrix

from tensorflow.keras import Model

def model_maker_model(DATASET_PATH):

  '''Классификатор из коробки медиапап.

    Принимае пути к датасету. Внутри детектить руки, находит 21 точку (63 признака)
    по ним строит классификацию
  '''

  name = "model_maker"

  data = gesture_recognizer.Dataset.from_folder(
    dirname=DATASET_PATH,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
  )
  train_data, rest_data = data.split(0.8)
  validation_data, test_data = rest_data.split(0.5)

  hparams = gesture_recognizer.HParams(export_dir="exported_model",
                                     epochs = 50,
                                     shuffle = True)
  options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
  model = gesture_recognizer.GestureRecognizer.create(
      train_data=train_data,
      validation_data=validation_data,
      options=options
  )

  loss, acc = model.evaluate(test_data, batch_size=1)
  print(f"Test loss:{loss}, Test accuracy:{acc}")

  return [name, loss, acc]

def build_model_conv2D():

    '''Классификатор на CNN.
    Проходим 2мя блоками свертки, после вытягиваем все  Флатеном в один вектор
    пропускаем через линейные слои и LeakyReLU, чтоб даже слабые сигналы на малом датасете не затухали
    '''

    model = Sequential()

    model.add(Input(shape=[224, 224, 3]))

    model.add(Conv2D(32, 3, padding='same', strides=2, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(32, 2, padding='same', strides=2, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  )
    return model

def model_conv2D_eval(model,data):
    '''Функция обучения модели.

    Принимает
    model – модель
    data – список параметров для распаковки. Ввиду большого кол-во значений выбран такой способ передачи выборок

    Возвращает:
    history – список метрик после обучения моедли

    После заверешния оубчения печатает:
      print(f"Test loss:{loss}, Test accuracy:{acc}") – тестовые лосс и точность

      print_confusion_matrix_details(cm, ['other', 'empty', 'money']) – матрицу ошибок по классам
                                                                        (не нравится plt, сам текстом рисую)
    '''

    X_images_train = data[0]
    X_images_val = data[1]
    X_images_test = data[2]
    y_train = data[3]
    y_val = data[4]
    y_test = data[5]

    model.summary()

    history = model.fit(X_images_train, y_train,
                        validation_data=(X_images_val, y_val),
                        epochs=50,
                        batch_size=32)

    loss, acc = model.evaluate(X_images_test, y_test, batch_size=32)
    print()
    print(f"Test loss:{loss}, Test accuracy:{acc}")

    y_pred_proba = model.predict(X_images_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    print()
    print_confusion_matrix_details(cm, ['other', 'empty', 'money'])

    return history


def build_conv2d_and_coords_model():

    '''Гибридный классификатор на CNN и линейных слоях:

        conv2d_block – проходим 2мя блоками свертки, после вытягиваем все  Флатеном в один вектор

        dense_block – протягиваем данные через линейный слой, LeakyReLU для сохранения тихих сигналов

        concat – склеиваем conv2d_block после Флатен и dense_block

        combined – пропускаем через линейные слои для обработки concat

        Результат combined классифицируем
        '''

    img_input = Input(shape=[224, 224, 3])
    coord_input = Input(shape=88,)

    conv2d_block = Conv2D(32, 3, padding='same', strides=2, activation="relu")(img_input)
    conv2d_block = BatchNormalization()(conv2d_block)
    conv2d_block = MaxPooling2D()(conv2d_block)

    conv2d_block = Conv2D(32, 2, padding='same', strides=2, activation="relu")(conv2d_block)
    conv2d_block = BatchNormalization()(conv2d_block)
    conv2d_block = MaxPooling2D()(conv2d_block)

    conv2d_block = Dropout(0.5)(conv2d_block)
    conv2d_block = Flatten()(conv2d_block)

    dense_block = Dense(64)(coord_input)
    dense_block = LeakyReLU(alpha=0.01)(dense_block)
    dense_block = BatchNormalization()(dense_block)
    dense_block = Dropout(0.5)(dense_block)

    concat = Concatenate()([conv2d_block, dense_block])

    combined = Dense(32, activation="relu")(concat)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)

    combined = Dense(16, activation="relu")(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)

    output = Dense(3, activation="softmax")(combined)

    model = Model(inputs=[img_input, coord_input], outputs=output)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  )

    return model

def eval_conv2d_and_coords_model(model,data):
    '''Функция обучения модели.

    Принимает
    model – модель
    data – список параметров для распаковки. Ввиду большого кол-во значений выбран такой способ передачи выборок

    Возвращает:

    history – список метрик после обучения моедли

    После заверешния оубчения печатает:

    print(f"Test loss:{loss}, Test accuracy:{acc}") – тестовые лосс и точность

    print_confusion_matrix_details(cm, ['other', 'empty', 'money'])
    '''

    X_images_train = data[0]
    X_images_val = data[1]
    X_images_test = data[2]
    X_coord_train = data[3]
    X_coord_val = data[4]
    X_coord_test = data[5]
    y_train = data[6]
    y_val = data[7]
    y_test = data[8]
    model.summary()

    history = model.fit([X_images_train, X_coord_train], y_train,
                        validation_data=([X_images_val, X_coord_val], y_val),
                        epochs=50,
                        batch_size=32)

    print()
    loss, acc = model.evaluate([X_images_test, X_coord_test], y_test)
    print(f"Test loss:{loss}, Test accuracy:{acc}")

    y_pred_proba = model.predict([X_images_test, X_coord_test])
    y_pred = np.argmax(y_pred_proba, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    print()
    print_confusion_matrix_details(cm, ['other', 'empty', 'money'])

    return history