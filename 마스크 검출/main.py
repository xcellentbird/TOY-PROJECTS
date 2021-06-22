from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 영상(이미지)에서 얼굴을 찾아내는 모델이에요.
facenet = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# 이거는 얼굴 영상을 이용하여 마스크를 쓴 건지, 안 쓴 건지 판단하게 해주는
model = load_model('mask_detector')

# 컴퓨터 카메라로부터 이미지를 받아오는 녀석
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 영상을 읽어옵니다.
    ret, img = cap.read()
    if not ret:
        break
    
    # 영상의 높이 너비를 가져와 저장
    h, w = img.shape[:2]

    # 이미지 normalize and face detect
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    facenet.setInput(blob)
    dets = facenet.forward() # 이 함수를 쓰면 얼굴의 위치값이 반환된다.

    result_img = img.copy()

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2] # 얼굴인지 아닌지 확률값을 가져온다.
        if confidence < 0.5: #얼굴일 확률이 0.5 이하라면 아래 코드 무시
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        
        face = img[y1: y2, x1: x2] # 이미지에서 얼굴만 잘라낸다.
        if face.size == 0: # 얼굴 영상의 크기가 0일 경우 무시
            continue
        face_input = cv2.resize(face, dsize=(128, 128)) # 얼굴 영상(이미지)의 크기를 224 * 224로 바꾸고
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB) # RGB배열로 바꾼다.
        #face_input = preprocess_input(face_input) # 이미지를 전처리를 하고
        face_input = np.expand_dims(face_input, axis=0) # 이미지 입력을 위해서 차원을 하나 더 생성 3 * 224 * 224 => 1 * 3 * 224 * 224

        pred = model.predict(face_input).squeeze() # 얼굴 이미지에서 마스크 쓴 확률, 안 쓴 확률을 예측해내고, squeeze함수를 통해서 차원을 줄여준다.

        incorrect_mask, mask, nomask= pred # 각 클래스의 확률반환.
        print('incorrect mask: {:.3f}%,  mask: {:.3f}%,  unmask: {:.3f}%'.format(incorrect_mask * 100, mask * 100, nomask * 100))
        
        # 클래스의 확률을 이용해서 적절하게 알고리즘 작성하시면 됩니다.
        max_class = np.argmax(pred)
        if nomask > 0.3: 
            color = (0, 255, 0)
            label = 'No Mask %d%%' % (nomask * 100) # 라벨 값(str type)을 마스크 ** % 로 설정
        elif incorrect_mask > 0.1:
            color = (0, 0, 255)
            label = 'Incorrect Mask %d%%' % (incorrect_mask * 100)
        else:
            color = (255, 0, 0)
            label = 'Mask %d%%' % (mask * 100)
        
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA) # 원 이미지에 사각형을 그려넣는다.
        # 원 이미지에 텍스트를 삽입한다.
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('result', result_img)
    #cv2.imshow('face', face)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
