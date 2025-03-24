import cv2
import numpy as np

# 하르 특징 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 샤프닝 커널 정의
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])


# 모자이크 함수 정의
def apply_mosaic(image, factor=10):
    # factor 모자이크 사이즈
    # 이미지의 크기 얻기
    height, width = image.shape[:2] # shape에서 w,h 값만 가져오겠다

    # 모자이크 효과를 위해 작은 크기로 이미지를 리사이즈
    small = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_LINEAR)

    # 다시 원래 크기로 리사이즈하여 모자이크 적용
    mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    return mosaic


# 이미지 읽기
image = cv2.imread('test_IMG.jpg')  # '.jpg'를 원하는 이미지 파일로 교체하세요.

# 그레이스케일로 변환 (얼굴 검출을 위해)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 모자이크 처리된 이미지 복사본 만들기
mosaic_image = apply_mosaic(image)

for (x, y, w, h) in faces:
    # 얼굴 영역 잘라내기
    face_region = image[y:y + h, x:x + w]

    # 샤프닝 필터 적용
    sharpened_face = cv2.filter2D(face_region, -1, sharpening_kernel)

    # 필터링된 얼굴을 원본 이미지에 덮어쓰기
    image[y:y + h, x:x + w] = sharpened_face

    # 모자이크 처리된 이미지를 원본 이미지에 덮어쓰기 (얼굴 영역 제외)
    mosaic_image[y:y + h, x:x + w] = image[y:y + h, x:x + w]

# 결과 영상 출력
cv2.imshow('Face Sharpening with Mosaic', mosaic_image)

# 'q'를 누르면 종료
cv2.waitKey(0)
cv2.destroyAllWindows()