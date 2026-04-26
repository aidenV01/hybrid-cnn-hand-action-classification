import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import gdown
import os

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
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

'''Константы путей к модели и датасету'''
MODEL_PATH = "/content/hand_landmarker.task"
DATASET_PATH = "/content/Dataset"


def print_confusion_matrix_details(cm, class_names):
    '''Функция вывода матрицы ошибок'''
    print("\nConfusion matrix (rows: true, cols: predicted):")
    for i, true_name in enumerate(class_names):
        row = cm[i]
        correct = row[i]
        errors = []
        for j, pred_name in enumerate(class_names):
            if i != j and row[j] > 0:
                errors.append(f"{row[j]} как {pred_name}")

        error_str = ", ".join(errors) if errors else "нет ошибок"
        print(f"{true_name:10} → {row}   ({correct} угадал, {error_str})")

'''Инициализация и настрйока модели'''
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = MODEL_PATH),
    running_mode = VisionRunningMode.IMAGE)

def collect_coords(dataset_path, options):

    '''Функция для сбора коориднат

    На вход принимает путь к датасету и настройки модели

    На выходе отдает:
    np.array(X) – массив координат (21 точка ладони, каждая точка x,y,z, всего 63 координаты
    np.array(y) – массив лейблов закодированных в 0, 1, 2
    paths – список с путями файлов, необходим для последующей передачи в CNN'''

    X = []
    y = []
    skipped = 0
    paths = []

    label_map = {
        "empty": 1,
        "money": 2,
        "other": 0
    }

    with HandLandmarker.create_from_options(options) as landmarker:

        for folder in os.listdir(dataset_path):

            folder_path = os.path.join(dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            label = label_map.get(folder)
            if label is None:
                continue

            for file in os.listdir(folder_path):

                file_path = os.path.join(folder_path, file)

                mp_image = mp.Image.create_from_file(file_path)
                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    skipped += 1
                    continue

                landmarks = result.hand_landmarks[0]

                palm = landmarks[0]
                palm_x, palm_y = palm.x, palm.y

                points = []

                for point in landmarks:
                    norm_x = point.x - palm_x
                    norm_y = point.y - palm_y
                    points.append([norm_x, norm_y, point.z])

                X.append(points)
                y.append(label)
                paths.append(file_path)

    print(f"Пропущено: {skipped}")

    return np.array(X), np.array(y), paths

'''Константы'''
fingers_tips = [4, 8, 12, 16, 20] #Индексы точек кончиков пальцев
base_joints = [1,5,9,13,17] #Индексы точек основания пальцев

def euclid_dist(coords, fingers_tips):

    '''Функция для рассчета эвклидового растояния между кончиками пальцев (точки 4, 8, 12, 16, 20)
    и основанием ладони (точка 0)

    Принимает:
    coords – список координат (Х)
    fingers_tips – список индексов кончиков пальцев

    Отдает:
    np.array(all_e) – список эвклидовых растояний'''

    all_e = []

    for sample in coords:

        e = []

        for tip in fingers_tips:

            d = np.linalg.norm(sample[tip] - sample[0])
            e.append(d)

        all_e.append(e)

    return np.array(all_e)

def cosine_dist(coords, fingers_tips, base_joints):

    '''Функция для рассчета косинуских растояний между пальцами

    Принимает:
    coords – список координат (Х)
    fingers_tips – список индексов кончиков пальцев (точки 4, 8, 12, 16, 20)
    base_joints – список индексов основания пальцев (точки 1, 5, 9, 13, 17)

    Отдает:
    np.array(all_cos) – список косинуских растояний'''

    all_cos = []

    for sample in coords:

        finger_vectors = []

        for tip, base in zip(fingers_tips, base_joints):
            v = sample[tip] - sample[base]
            finger_vectors.append(v)

        cos_features = []

        for i in range(len(finger_vectors)):
            for j in range(i+1, len(finger_vectors)):

                v1 = finger_vectors[i]
                v2 = finger_vectors[j]

                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_features.append(1 - cos)

        all_cos.append(cos_features)

    return np.array(all_cos)

def angle_features(coords, fingers_tips, base_joints):

    '''Функция для рассчета углов между пальцами

       Принимает:
       coords – список координат (Х)
       fingers_tips – список индексов кончиков пальцев (точки 4, 8, 12, 16, 20)
       base_joints – список индексов основания пальцев (точки 1, 5, 9, 13, 17)

       Отдает:
       np.array(all_angle_features) – список углов'''

    all_angle_features = []

    for sample in coords:

        finger_vectors = []

        for tip, base in zip(fingers_tips, base_joints):
            v = sample[tip] - sample[base]
            finger_vectors.append(v)

        angles = []

        for i in range(len(finger_vectors)):
            for j in range(i + 1, len(finger_vectors)):

                v1 = finger_vectors[i]
                v2 = finger_vectors[j]

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                cos_sim = np.dot(v1, v2) / (norm1 * norm2 + 1e-8)

                angle = np.arccos(np.clip(cos_sim, -1.0, 1.0))

                angles.append(angle)

        all_angle_features.append(angles)

    return np.array(all_angle_features)

'''Обработка координат.

Изначальное – 63 точки x, y, z (признаки)

после euclid_dist – 68 признаков
после cosine_dist – 78 признаков
после angle_features – 88 признаков'''

coords, y, image_paths = collect_coords(DATASET_PATH, options)

coords_flat = coords.reshape(coords.shape[0], -1)

e = euclid_dist(coords, fingers_tips)
c = cosine_dist(coords, fingers_tips, base_joints)
a = angle_features(coords, fingers_tips, base_joints)

X_cords = np.hstack([coords_flat, e, c, a])

def load_all_images(image_paths, target_size=(224, 224)):

    '''Функция загрузки изображений.

    Принимает:

    image_paths – путь к изображениям
    target_size – статичен

    Отдает:
    np.array(images) –изображения ввиде нампай массива'''

    images = []

    for path in image_paths:

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32)
        images.append(img)

    return np.array(images)

'''Загрзка всех изображений в константу '''
X_images = load_all_images(image_paths)

'''Разделение выборок'''
X_images_temp, X_images_test, X_cords_temp, X_cords_test, y_temp, y_test = train_test_split(X_images, X_cords, y,
                                                                                              test_size = 0.2,
                                                                                              stratify = y,
                                                                                              random_state = 42)

X_images_train, X_images_val, X_cord_train, X_cord_val, y_train, y_val = train_test_split(X_images_temp, X_cords_temp, y_temp,
                                                                                            test_size = 0.25,
                                                                                            stratify = y_temp,
                                                                                            random_state = 42)

'''Упаковка для более комфортной передачи в функции оубчения модели'''
data_for_conv2d = [X_images_train, X_images_val, X_images_test, y_train, y_val, y_test]
data_for_conv2d_and_coords = [X_images_train, X_images_val, X_images_test, X_cord_train,
                              X_cord_val, X_cords_test, y_train, y_val, y_test]
