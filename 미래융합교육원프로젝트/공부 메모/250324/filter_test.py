# 25.03.24
import numpy as np
import cv2

# 1. 기본 cv2에서 제공하는 필터
# 2. 자체적인 필터 정의 : kernel > 이미지 사이즈와 동일하거나 or 3 x 3 ~(홀수 단위)의 커널 생성 > np.array로 생성
# 3. ndarray 수치화된 이미지 [[255,220,170,~~~~~~~ 기존 이미지를 위해 새로운 데이터 레이어(전체 좌표에 대한 or 일부 좌표에 대한) 들리는 필터]]


# image=cv2.imread('s.jpg')
# kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) # 필터 -1 흐리게
#
# filter_image=cv2.filter2D(image,-1,kernel) # -1 기본 값(픽셀 값), RGB
#
# cv2.imshow('filter',filter_image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 데이터 필터?
# 계속 움직이며 휘는 필터링
# 모자이크 , 눈만 찾아서 확대 등의 예시

# 얼굴만 선명하게?

# 하르에서 사각형 영역을 따라 얼굴 부분만 검출
# 기본적인 하르 제작 후 필터링 제작 후
#

cap = cv2.imread('test_IMG5.webp')

# 이미지 크기와 높이 추출
height, width = cap.shape[:2]

# 모자이크 처리
factor = 7
s = cv2.resize(cap, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)  # 모자이크
m = cv2.resize(s, (width, height), interpolation=cv2.INTER_LINEAR)

# 얼굴 검출을 위한 CascadeClassifier 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 그레이스케일 변환
gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 검출된 얼굴 영역에 원본 이미지를 덮어씌움
for (x, y, w, h) in faces:
    face_area = cap[y:y + h, x:x + w]
    m[y:y + h, x:x + w] = face_area

# 결과 출력
cv2.imshow('filter', m)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 우선 여기까진 필터 적용 나머지 부분들 모자이크 법



























