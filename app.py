import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import layers, models

# 이미지 크기 및 경로 설정
img_size = (640, 640)

# 각 부위별로 모델 정의
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


# 각 부위별 모델 생성 및 학습
num_classes = len(head_train_generator.class_indices)
head_model = create_model()
head_model.fit(head_train_generator, epochs=40, validation_data=head_val_generator)

num_classes = len(eyes_train_generator.class_indices)
eyes_model = create_model()
eyes_model.fit(eyes_train_generator, epochs=50, validation_data=eyes_val_generator)

num_classes = len(ear_train_generator.class_indices)
ear_model = create_model()
ear_model.fit(ear_train_generator, epochs=20, validation_data=ear_val_generator)

# Streamlit 앱에서 모델 저장 및 로드를 위한 경로
head_model_path = 'head_model.h5'
eyes_model_path = 'eyes_model.h5'
ear_model_path = 'ear_model.h5'

# 모델 저장
head_model.save(head_model_path)
eyes_model.save(eyes_model_path)
ear_model.save(ear_model_path)
