# 
import cv2

# print(cv2.__version__) # open 비젼체크
# 
# cantrue = cv2.VideoWriter(0) # 웹캡 검사
# 
# cantrue.set(cv2.CAP_PROP_FRAME_WIDTH,340) # 캡쳐 프레임에 대한 width 설정 set
# cantrue.set(cv2.CAP_PROP_FRAME_HEIGHT,480)# 캡쳐 프레임에 대한 height 설정 set
# 
# while cv2.waitKey(33)<0: # 시용자가 키 입력 대기 시간을 33ms 로 설정하지 않고
#      키 이벤트가 발생하지 않으면 -1 반환하기에 <0 조건식으로 구성함
#  영상의 프레임 수 제어를 원할 떄 waitkey의 시간만을 이용해서 간단하게 프레임 수 제어 가능
#  ex) 초당 60 프레임 여상을 30 프레임으로 설정하려면 0.5뱃속
#  33으로 두면 대락 0.5배속
#     ret,fram=cantrue.read() # 캡쳐 액션 객체의 frame 전부 읽기
#     cv2.imshow("",fram) # fram 출력
# 
# cantrue.release()
# cv2.destroyAllWindows()

# import cv2
#
# # print(cv2.__version__)  # OpenCV 버전 확인
#
# cantrue = cv2.VideoCapture(0)  # 웹캠 연결 검사
#
# # 웹캠 캡쳐 프레임 설정
# cantrue.set(cv2.CAP_PROP_FRAME_WIDTH, 340)  # 캡처 프레임의 width 설정
# cantrue.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처 프레임의 height 설정
#
# while cv2.waitKey(33) < 0:  # 사용자가 키 입력 대기 시간을 33ms로 설정, 키 이벤트가 발생하지 않으면 -1 반환
#     # 영상의 프레임 수 제어를 원할 때 waitKey의 시간을 이용해서 간단하게 프레임 수를 제어할 수 있습니다.
#     # 예) 초당 60 프레임 영상을 30 프레임으로 설정하려면 waitKey의 시간을 33으로 설정하면 됩니다.
#     # 33ms를 설정하면 약 30fps에 해당하고, 16ms를 설정하면 약 60fps에 해당합니다.
#     ret, frame = cantrue.read()  # 캡쳐 객체에서 프레임 읽기
#     cv2.imshow("s.jpg", frame)  # 프레임 출력
#
# cantrue.release()  # 웹캠 해제
# cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기


# image=cv2.imread("s.jpg",cv2.IMREAD_ANYCOLOR)
# cv2.imread()의 읽기 속성에 지정가능한 속성 리스트

#cv2.imread("s.jpg",cv2.IMREAD_ANYCOLOR) : 원본
# cv2.IMREAD_GRAYSCALE : 흑백 (1채널) 그레이스케일
# cv2.IMREAD_COLOR : 3체널의 RGV 이미지
# cv2.IMREAD_ANYDEPTH : 이미지에 따라 정밀도를 16/32/8 바트로 사용
# cv2.IMREAD_ANYCOLOR : 가능한 3 채널 색상 이미지로 사용
# cv2.IMREAD_REDUCED_GRAYSCALE_2 : 1 채널 흑백 1/2 크기
# cv2.IMREAD_REDUCED_GRAYSCALE_4 : 1 채널 흑백 1/4 기
# cv2.IMREAD_REDUCED_GRAYSCALE_8 : 1 채널 흑백 1/8 크기
# cv2.IMREAD_REDUCED_COLOR_2 : 3 채널 BGR 1/2 크기
# cv2.IMREAD_REDUCED_COLOR_4  : 3 채널 BGR 1/4 크기
# cv2.IMREAD_REDUCED_COLOR_8 : 3 채널 BGR 1/8 크기


# 동영상의 반복 재생
# capture=cv2.VideoCapture("~~~.mp4")
# while cv2.waitKey(33):
#     if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
#         # 위 조건문은 프레임 카운트와 연재 프레임이 같다면 set으로 프레임을 처음으로 지정함
#         capture.set(cv2.CAP_PROP_POS_FRAMES,0)
#         # 0 프레임으로 지정해서 반복 재생 : 영상 무한 재생
#
#
#         ret,frame=capture.read()
#         cv2.imshow("fr",frame)
#
# capture.release()
# cv2.destroyAllWindows()


# src= cv2.imread("s.jpg",cv2.IMREAD_ANYCOLOR) # cv2의 서드로 읽은 이미지는 ndarray
# print(type(src))
# height,width,channel=src.shape
# matrix = cv2.getRotationMatrix2D((width/2,height/2),90,2) # 각도,배율, 중심점 좌표
# # getRotationMatrix2D
# # center 좌표(튜플)/ 회전 각도/ 확대배율 설정

# des=cv2.warpAffine(src,matrix,(width,height)) # src의 matrix를 적용한 것(필터)
# #  아핀 변환 함수 : 원본 이미지의 아핀 맵 행렬을 적용하고 출력 이미지 크기(dsize)로 변형해서 변환
#
# cv2.imshow("src",src)
# cv2.imshow('des',des)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 이미지 확대 축소
# src= cv2.imread("s.jpg",cv2.IMREAD_ANYCOLOR)
# height,width,channel = src.shape
#
# dst= cv2.pyrUp(src,dstsize=(width*2,height*2),borderType=cv2.BORDER_DEFAULT)
# dst2= cv2.pyrDown(src)
# # 보더 타입
# # BORDER_REFLECT_101 (기본값): 가장자리를 반사해서 새로운 픽샐을 채움
# # BORDER_REPLICATE : 가장자리 픽샐을 복사해서 채움
# # BORDER_CONSTANT 지정된 성상수(예를 들어 0또는 검정)으로 픽셀을 채움
#
# cv2.imshow('src',src)
# cv2.imshow('dst',dst)
# cv2.imshow('dst2',dst2)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # 업 샘플링 : 원본 이미지 확대
# # 다운 샘플링 : 원본 이미지 축소
# # pyrUp으로 원본이미지를 2배 확대하는 경우 이미지의 크기는 원본 이미지의 크기의 2배 확정
# # 이때 원본 이미지의 총 가로 픽셀수가 200일 때 이미지의 가로 픽셀 수는 400이 됌
# # 따라서 이 상황에 존재하는 200 픽셀 외 200 픽셀을 어떻게 채울 것인가 > borderType으로 결정



# 영상 처리의 단계: 사이즈/ 잘라내기 / 확대 축소 > 이진화(그레이스케일 1채널)> 임계값 설정 > 경계선 edge 검증> 건투어 contours 추출

# grayImage=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# cv2.threshold(grayImage,임계값),cv2.THRESH_BINARY)

# src=cv2.imread('s.jpg',cv2.IMREAD_ANYCOLOR)
# gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# ret,frame = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
# # 50을 기준으로 50 초과하는 픽셀들을 255 처리 50이하 0 처리해서 이진화
# # cv2.THRESH_BINARY: 임계 값 초과하는 경우 maxval 로 아닐경우 0으로
# # cv2.THRESH_BINARY_INV: 임계 값 초과하는 경우 0 로 아닐경우 255으로
# # cv2.THRESH_TOZERO: 임계 값 초과하는 경우 변함 없음 아닐경우 0으로
#
#
# cv2.imshow('frame',frame)
# cv2.waitKey()
# cv2.destroyAllWindows()


#이미지 블러 처리
#블러는 영상의 샤프니스를 줄여서 노이즈를 없애거나 외부 영향을 최소화 하는 목적으로 사용
#블러는 영상이나 이미지를 번지게 하고, 해당 픽셀의 주변 값과 비교하고 계산해서 픽셀들의 색상을 재조정

# src=cv2.imread("s.jpg",cv2.IMREAD_ANYCOLOR)
# dst=cv2.blur(src,(9,9),anchor=(-1,-1),borderType=cv2.BORDER_DEFAULT)
# cv2.imshow("blur",dst)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 앵커
# 고정점
# 커닝을 통해 건별투선된 값을 할당한 지점
# 건별투선이란 새로운 픽셀을 만들어 내기 위해 커닝 크기의 최고 값을 이용해서 어떤 시스템을 통화해 계산하는 것을 의미
# 커널 내에서 고정점은 하나의 지점만을 가진다
# anchor(-1,-1) 따라 알아서 고정점 지정
# 보톤 앵코는 -1,-1로 사용

# 보더 타입 
# 가장자리 생성 방법
# 건별투선 적용시 이미지의 가장자리 처립 아식
# 건별투선을 적용하면 이미지의 가장자리 부분은 계산이 불가능함  이문제를 해결하려고 테두리 의 이미지 바깥쪽 가상의 픽셀을 만들어서 처리함
# 가상 픽셀의 값을 0으로 처리하거나 양의 값을 할당하거나 , 커널이 연산할 수 있는 부분부터 연산을 수행하기로 한다


# BORDER_CONSTANT : nnnn|ABCD |nnnn
# BORDER_REFLECT : DCBA|ABCD |DCBA
# BORDER_DEFAULT : 반사와 동일, 기본 값
# BORDER_TRANSPARENT : 경계 투명 생성

# src=cv2.imread('s.jpg',cv2.IMREAD_ANYCOLOR)
# gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# sobel=cv2.Scharr(gray,cv2.CV_8U,1,0,3)
# laplacianA=cv2.Laplacian(gray,cv2.CV_8U,ksize=3)
#
# canny=cv2.Canny(src,100,255)
# cv2.imshow("sobel mask",sobel)
# cv2.imshow("lapla",laplacianA)
# cv2.imshow("canny",canny)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 엣지 검출 -  소별마스크
# 소별 마스크를 통해 경계성 검출 기능
# 인접한 픽셀들의 차이로 기울기 크기를 통한 검출
# 소별마스크 함수의 파라미터
# dst = cv2.Sobel(src,ddepth,dx,dy,ksize,scale,delta,borderType)
# ddepth 정밀도
# dx:  x방향 계수 > x 미눕ㄴ치수 : d이미지 수평 방향 가울기 계산
# dy:  y방향 계수 > y 미분치수 : 이미지 수직 방향 기울기 계산
# ksize  커널 사이즈 소별 마스크의 크기( 모든 커널은 홀수로 지정 최대 31까지)
# scale 비율
# delta: 옵셋
# borderType: 테두리 외삼법 플래그


# 라플라시안 함수
# 라플라시안은 2차 미분 형태로 가장자리가 밝은 부분부터 발생한 것인지 아니면 어두운 부분부터 발생한 것ㅇ니지
# 알 수있음
# dst = cv2.Laplacian(src,ddepth,ksize)
# 이미지 정밀도 (ddepth)
# ksize: 커널 사이즈
# cv2.CV_8U : w정밀도 중 8비트 부호 없는 정수 > 0~255
# 각 픽셀 값을 0~55 사이 수치로 표현하는 체계
# RGB 페계 표현에서 CV_8U 씀

# src = cv2.imread('s.jpg',cv2.IMREAD_ANYCOLOR)
# gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# ret,binary=cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
# binary=cv2.bitwise_not(binary)
# contours,hierarchy = cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# cv2는 컨투어 찾기 함수는 이전 이미지에서 윤곽선을 검색함
# fincontour의 매개변수 : 대상 이미지/ 검색 방법/ 근사화방법
# 반환값은 윤곽선과 계층구조
# 윤곽선은 넘파이 배열로 담아서 리턴옴
# 계층 구조는 윤곽선의 등고선 레이어
#
# cv2.RETR_CCOMP :  계층 구조를 2단계로 구성
# cv2.CHAIN_APPROX_NONE : 윤곽선의 모든 점을 반환
#
# for i in range(len(contours)):
#     cv2.drawContours(src,[contours[i]],0,(0,255,0),1) # 색깔은 b,g,r 순서
#     # print(i,hierarchy[0][i]) # 계층 구조 값 확인용, i는 몇번째 컨투어
#     cv2.imshow('src',src)
# cv2.waitKey()
# cv2.destroyAllWindows()

#  적응형 임계값 - 어댑티드 스레시 홀드
# 적으용형 이진화 알고리즘은 입력 이미지에 따라 임계값이 스스로 다른 값을 할당하도록 구성된 이진화 알고리즘
# 단순 스레시홀드 함수 대비 경계를 잘 잡아냄
# 이미지에 따라 어떤 입계값을 줘도 애매한 이미지들이 있음( 너뮤 다양한 값들이 쓰였거나 이미지가 애매한)
# 혹은 조명이 특정 부위에 강하게 있거나 반사가 심하거나 등
#  극 소적으로 다른 명계값을 적용해야만 경계를 검출할 수 있는  이미지들이 많음
#  이런 경우에는 어뎁티드 스레시홀드를 쓰는 게 나음


# src = cv2.imread('s.jpg')
# gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,467,100)
# #  매개변수
# # maxvalue 255 : 최대 값 255
# # cv2.ADAPTIVE_THRESH_MEAN_C : 이진화할 각 영역의 픽셀 값에 대해 그 영역의 평균으로 임계값을 계산하는 방식의ㅣ 플러그
# # cv2.THRESH_BINARY: 기본적인 이진화 방식을 채택
# # 467값 : addptive threshold 함수에서 각 지역의 임계값을 계산 시 사용할 아웃 영역 크기 커널 사이즈
# # C 값 37:  어뎁티드 스레시홀드에서 각 지역의 평균값을 보강값으로 봐주는 값 수치
# # C 값이 클수록 더 여러 픽셀이 검은색(0)으로 이진화됌
#
# cv2.imshow('bin',binary)
# cv2.waitKey()
# cv2.destroyAllWindows()



# 이미지 침식과 팽창
# 이미지의 침식(erosion)은 경계를 축소(침식)시키는 효과, 작은 노이즈 제거 / 패딩 입히는 것처럼
# erores_image=cv2.erode(src,kernel=13,iterations=1)

# 이미지의 팽창(dilation)은 이미지 경계를 확장하는 효과 , 침식에서 축소된 객체의 복원 작업이나 작은 구멍 메우기
# dilate_image=cv2.dilate(image,kernel=13,iterations=1)

import numpy as np
# image=cv2.imread('s.jpg',0)
# kernel=np.ones((3,3),np.uint8)
# eroded_image=cv2.erode(image,kernel,iterations=1)
# dilate_image=cv2.dilate(image,kernel,iterations=1)
#
# cv2.imshow('origin',image)
# cv2.imshow('eroded',eroded_image)
# cv2.imshow('dilate',dilate_image)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
# 이미지 편집기 제작 가능
# 예를 들어서 키에 따라서 이미지들을 편집하는 것 이나 버튼을 통해서 하는 것(이미지 클릭 등)





























