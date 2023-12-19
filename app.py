import streamlit as st
import cv2
import pandas as pd
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

# 부위별 모델 생성
head_model = create_model()
eyes_model = create_model()
ear_model = create_model()

st.title("아동 그림 심리 분석")

# 파일 업로드
uploaded_file = st.file_uploader("이미지를 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 업로드한 이미지 표시
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=img_size)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    # 예측을 위한 이미지 전처리
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 각 모델로 예측 수행
    head_prediction = head_model.predict(img_array)
    eyes_prediction = eyes_model.predict(img_array)
    ear_prediction = ear_model.predict(img_array)

    # 결과 표시
    st.subheader("예측 결과:")
    st.write("머리 결과:", "자신감이 결여됐음을 나타냅니다." if head_prediction[0][0] > 0.5 else "교만한 혹은 과장된 자신을 나타내는 그림이지만, 만약 어린이라면 머리를 크게 그리는 경우가 많기 때문에 큰 의미를 갖진 않습니다.")
    st.write("눈 결과:", "세상을 보긴 하지만 자신을 내보이고 싶지는 않다는 것을 의미" if eyes_prediction[0][0] > 0.5 else "세상을 경계하고 불안감이 많은 사람의 눈이 큰 경우가 많습니다")
    st.write("귀 결과:", "청각에 문제가 있음을 표현할 수 있음" if ear_prediction[0][0] > 0.5 else "듣는 것을 상대적으로 중요하게 생각하지 않은 사람일 수 있으며, 고집스러움을 나타낼 수도 있습니다")
