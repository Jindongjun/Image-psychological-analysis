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

# 부위별 모델 불러오기
head_model = load_model(head_model_path)
eyes_model = load_model(eyes_model_path)
ear_model = load_model(ear_model_path)

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
    st.write("머리 결과:", "얼굴을 몸통보다 상대적으로 작게 그렸다는 것은 자신감이 결여됐음을 나타냅니다." if head_prediction[0][0] > 0.5 else "얼굴을 크게 그렸다는 것은 교만한 혹은 과장된 자신을 나타내는 그림이지만, 만약 어린이라면 머리를 크게 그리는 경우가 많기 때문에 큰 의미를 갖진 않습니다.")
    st.write("눈 결과:", "눈을 작게 또는 점이나 선으로 그렸다는 것은 세상을 보긴 하지만 자신을 내보이고 싶지는 않다는 것을 의미합니다" if eyes_prediction[0][0] > 0.5 else "세상을 경계하고 불안감이 많은 사람의 눈이 큰 경우가 많습니다")
    st.write("귀 결과:", "귀를 크게 그렸다는 것은 청각에 문제가 있음을 표현하고 있을 수 있습니다." if ear_prediction[0][0] > 0.5 else "귀를 작게 그렸다는 것은 듣는 것을 상대적으로 중요하게 생각하지 않은 사람일 수 있으며, 고집스러움을 나타낼 수도 있습니다")
