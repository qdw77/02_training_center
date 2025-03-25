import cv2
import numpy as np
import requests

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import dlib
#
#
# img_cap = cv2.imread('test_IMG5.webp') # 확인용 이미지2
# img_recap=img_cap.copy() # 출력용
#
#
# # HOG(Histogram of Oriented Gradient) + SVM로 사람 얼굴 인식
# detector =dlib.get_frontal_face_detector() # 얼굴 전체
# img_re2cap =cv2.cvtColor(img_recap,cv2.COLOR_BGR2RGB) # 색 RGB로
# img_caps = detector(img_re2cap,1) # img_re2cap 출력 이미지, 1은 해상도
#
# for cap_img in img_caps:
#     left = cap_img.left()
#     # 얼굴 영역 좌 상단 y
#     print("left: ",left)
#     top = cap_img.top()
#     # 얼굴 영역 좌 상단 x
#     print("top: ",top)
#     right = cap_img.right()
#     # 얼굴 영역 우 하단 y
#     print("right: ",right)
#     bottom = cap_img.bottom()
#     # 얼굴 영역 우 하단 x
#     print("bottom: ",bottom)
#     cv2.rectangle(img_recap,(left,top), (right,bottom), (0,255,0), 2, lineType=cv2.LINE_AA)
# show_img=cv2.cvtColor(img_recap,cv2.COLOR_BGR2RGB)
# plt.imshow(show_img)
# plt.show()

# 링크로
url = 'https://i.namu.wiki/i/FHFPEVZjGJuBsijqiVLxOij8R9J0EVK4wZzJZjZESBMbE-5kf0_CQ7WnPjbrTM4V_V3B3WxTR0ceMx_mly7V8Q.webp'
response = requests.get(url)
img_array = np.array(bytearray(response.content), dtype=np.uint8)

# 이미지를 OpenCV 형식으로 디코딩
cap = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
# cap = cv2.imread('test_IMG5.webp') # 확인할 이미지

# 이미지 크기와 높이 추출
height, width = cap.shape[:2] # height, width, channels(컬러 정보) 획득 가능 그중 height, width 저장
print("height: ",height," / width: ",width)

# 모자이크 처리
factor = 10 # 모자이크 크기
s = cv2.resize(cap, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)  # 모자이크 제작
m = cv2.resize(s, (width, height), interpolation=cv2.INTER_LINEAR) # 모자이크 범위
# s에서 모자이크를 만들어서 m에서 전체 범위를 지정
# interpolation > 보간법이라고 함, 보간법이란 알려진 두점 사이의 어느 지점의 값을 추정하는 것
# INTER_LINEAR 양선형 보간법 > 4(2x2) 이웃 픽셀을 사용해서 새로운 픽셀을 계산(생성), 가까운 이웃일 수록 영향력이 크고 효율이 좋으며 확대에 주로 사용

# 분류 CascadeClassifier , haarcascade_frontalface_default.xml 얼굴 인식 데이터 파일
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 얼굴 인식을 위한 색 변환
gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

# detectMultiScale 좌표 값 반환
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=5, minSize=(50, 50))
# scaleFactor > 값을 이용해서 이미지를 축소해가며 얼굴 값 찾기
# minNeighbors > 검출 될때 연속해서 5번 이상 검출 된 것만 출력(기본 값은 3)
# minSize > 최소 사이즈

# faces에서 x,y,w,h는
# x,y: 좌상단(시작점) / w: 폭 , h: 높이
# [y(시작점):y+h(높이), x(시작점):x+w(폭)]
# y부터 y+높이,x부터 x+폭
# y는 시작점이고 시작점 + 높이(h)하는 이유는 범위 설정 때문에 x도 마찬가지
# y,x는 시작할 지점, h,w는 구할 값
# y축 기준 점 부터 h까지(y:y + h),x축 기준점부터 w까지(x:x + w)

# 검출된 얼굴 영역에 원본 이미지를 덮어씌움
print(faces)
for (x, y, w, h) in faces:
    print("y: ",y," /  x: ",x," &  w: ",w," /  h: ",h)
    face_area = cap[y:y + h, x:x + w]
    # 원본 이미지에 얼굴 좌표를 face_area에 담고
    m[y:y + h, x:x + w] = face_area
    # face_area에 필터 적용
    # cv2.rectangle(m, (x, y), (x + w, y + h), (255, 0, 0), 2) # 얼굴 칸 확인

# 결과 출력
cv2.imshow('filter', m)
cv2.waitKey(0)
cv2.destroyAllWindows()