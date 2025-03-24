# 250324
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 하르 특징
# 이미지의 특정 패턴을 나타내는 수학적 규칙 기반 검출 방법임
# 1. 앳지 특징: 밝기가 갑자기 변하는 경계를 찾음
# 2. 라인 특징: 밝기 변하가 선형적인 영역을 찾음
# 3. 센터서리라운드 특징: 밝기가 주변과 비교하여 큰 변화가 있는 곳

# # 웹캠 대신 동영상 파일 사용
# cap = cv2.VideoCapture("test_people.mp4")

# cap=cv2.VideoCapture(0)
# while True:
#     ret,frame=cap.read()
#     if not ret:
#         break
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     faces=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
#
#     for(x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#     cv2.imshow('detection',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
