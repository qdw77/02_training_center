# 2025.04.07
from threading import activeCount

import matplotlib.pyplot as plt
# 커널, 수치 / in hidden out
# 신경막 NN
# 뉴런 네트워크 신경막


# 머신러닝
# 머신러닝은 데이터를 통해 알고리즘이 스스로 학습하도록 하는 것
# 데이터 패턴을 찾고 예측, 결정내림

# 기법
# 감독학습 수퍼바이스드 러닝(Supervised Learning)
# 비 감독 학습 인수바이어스드 러닝(Unsupervised Learning)
# 강화학습 레인포스 러닝(Reinforcement Learning)


# 머신러닝 상대적으로 작은 규모의 데이터나 간단한 모델(선형회귀, 의사결정 트리, KNN 등)





# 딥러닝
# 딥러닝은 인공신경망을 기반으로 하는 곳
# 뉴럴 네트워크를 사용하여 데이터 분석/예측

# 다층 신경망을 이용해서 데이터의 특성(f) 추출하고 대규모 데이터 셋과 복잡한 모델 다룸
# 음성처림/nip자여뉴 이미지 처리/이미지 처미)


# AI

# 파이썬에서의 딥러닝
# 텐서플로우 구글 개발 딥러닝
# 케리스 텐서플로우에서 제공하는 api
# 파이토치 페이스북에서 개발 딥러닝 libㄴ

# 파이썬에서 머신러닝
# 사이캇런 : 분류 회귀 등
# XGBoost
# LightGBM




































































# 선항 회귀
# 데이터의 독립변수와 종속변수 사이의 선형관계를 모델링하는 기법
# 예측값을 얻는 목적


# 사이킷런을 사용하여 머신러닝 활용
# 사이킷런은 분류 회귀 군집 다양한 알고리즘이 있다
# 주요 클래스 fit(학습), predict(예측), score(성능평가)로 구분

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression # 선향회귀 클래스 사용
# from sklearn.model_selection import train_test_split # 훈련셋 테스트셋
# from sklearn.datasets import make_regression # 회귀 데이터셋 만드는 함수
#
#
# # 1. 데이터의 생성
# # make_regression 함수는 회귀 예측 가상 데이터 생성
#
# X,Y = make_regression(n_samples=100,n_features=1,noise=20,random_state=1)
# # n_samples : 샘플의 수(데이터 수)
# # n_features : 각 데이터의 특성(독립변수의 수)
# # noise : 잡음을 넣어서 데이터 패턴의 불규칙성을 줌, 노이즈가 너무 크다면 학습이 어렵고(현실성을 줌)
# # random_state : 난수 생성 기준 값
#
#
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.2,random_state=1)
# # 트레인 테스트 스플릿이 학습용과 세스트용 데잍터로 나눠줌
# # 0.2 만큼의 테스트 데이터를 분할 나머지 80%는 학습용으로 사용 예정
# model=LinearRegression() #선향회귀 모델 객체선언
# # 위에서 만든 데이터셋을 model()에다가 넣어서 훈련시킬 예정
# # 모델학습은 fit으로 학습시킨다
# # fit()
# model.fit(X_train,Y_train)
# # 예측
# Y_pred=model.predict(X_test)
# plt.scatter(X_test,Y_test,color='blue',label='value')
# plt.plot(X_test,Y_pred,color='red',linewidth=2,label='pred')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.show()
#
# print(model.score(X_test,Y_test))
# # 경청갯수: 모델이 얼마나 잘 추측하는가 0~1 사이의 값(정확도) , 0.98 이상이면 정확도 높음

# KNN K nearest neighbor
# 최근점 이웃 알고리즘
# 머신러닝 사이킷런

# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

# np.random.seed(1)
# # 남녀의 키, 체중 데이터 생성
# h_w=np.random.normal(160,5,50) # 키 평균 170 표준편차 5, 50개
# w_w=np.random.normal(50,5,50)
#
# h_m=np.random.normal(175,7,50)
# w_m=np.random.normal(70,7,50)
#
# X=np.vstack((np.column_stack((h_w,w_w)),np.column_stack((h_m,w_m))))
# y=np.array([0]*50+[1]*50) # 여성 0, 남성 1로 레이블로(정답1)
#
# X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)
# Knn=KNeighborsClassifier(n_neighbors=3)
# Knn.fit(X_train,Y_train)
# y_pred=Knn.predict(X_test)
# accuracy=accuracy_score(Y_test,y_pred)
# print(accuracy)
#
# plt.scatter(X_train[:,0],X_train[:,1],c=Y_train,marker='o',label='train')
# plt.scatter(X_test[:,0],X_test[:,1],c=Y_test,marker='x',label='test')
# x_min,x_max=X[:,0].min() -1,X[:,0].max() +1
# y_min,y_max=X[:,1].min() -1,X[:,1].max() +1
# xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
# Z=Knn.predict(np.c_[xx.ravel(),yy.ravel()])
# Z=Z.reshape(xx.shape)
#
# plt.contourf(xx,yy,Z,alpha=0.3)
# plt.legend()
# plt.xlabel('h')
# plt.ylabel('w')
# plt.show()


# x=np.array([[2,88],[3,90],[5,85],[1,65],[4,95],[6,85],[2,75],[7,90],[3,80],[8,90]])
# y=np.array([0,1,1,0,2,2,2,0,1,2])
# # 0 1 2 순으로 높음
#
# X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1)
# model=KNeighborsClassifier(n_neighbors=3) # 판단 갯수
# model.fit(X_train,Y_train)
# Y_pred=model.predict(X_test)
# print(classification_report(Y_test,Y_pred))


# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# cat_size=np.random.randint(30,51,333)
# cat_weight=np.random.randint(3,9,333)
# caat_legs = np.ones(333)*4
# cat_ear_shape=np.zeros(333)
# cat_food=np.zeros(333)
# X_cat=np.column_stack((cat_size,cat_weight,caat_legs,cat_ear_shape,cat_food))
#
# elephant_size = np.random.randint(200,301,333)
# elephant_weight = np.random.randint(4000,6001,333)
# elephant_legs = np.ones(333)*4
# elephant_ear_shape=np.ones(333)
# elephant_food=np.ones(333)
# X_elephant=np.column_stack((elephant_size,elephant_weight,elephant_legs,elephant_ear_shape,elephant_food))
#
# dog_size = np.random.randint(100,301,334)
# dog_weight = np.random.randint(10,70,334)
# dog_legs = np.ones(334)*4
# dog_ear_shape=np.zeros(334)
# dog_food=np.zeros(334)
# X_dog=np.column_stack((dog_size,dog_weight,dog_legs,dog_ear_shape,dog_food))
#
#
# x=np.vstack((X_cat,X_elephant,X_dog))
# y=np.array([0]*333+[1]*333+[2]*334)
#
# X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#
# scaler=StandardScaler()
# # 특점 스케일링 (표준화) 데이터 평균점 0 표준편차 1로 춘
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# model=MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000,random_state=1)
# # 다중퍼셉트론 10개ㄴ 뉴런의 2층구조(은닉층) maxiter 최대반복수
# model.fit(X_train_scaled,Y_train)
# y_pred = model.predict(X_test_scaled)
# accuracy=accuracy_score(Y_test,y_pred)
# print(f"모델 정확도: {accuracy:.5f}")
#
# size=float(input("크기 얼마?"))
# weight=float(input("무게 얼마?"))
# legs=float(input("다리 몇개?"))
# ear=float(input("귀 모양 어떰? 0작음, 1큼?"))
# food=float(input("밥 뭐먹음? 0 사료, 1물"))
#
# user_data=np.array([[size,weight,legs,ear,food]])
# user_data_scaled = scaler.transform(user_data)
# prediction = model.predict(user_data_scaled)
#
# if prediction ==0:
#     print("고양이인듯")
# elif prediction ==1:
#     print("코끼리 인듯")
# else:
#     print("강아지 인듯")


# 가충치
# y=wx+b
# y=wx 제곱+b

# ai 모델 신경망은 학습 과정을 통해 가중치를 조정한다
# 조정하는 과정은 경사하강법을 통해 이루어짐
# 초기화: 모델 처음 만들 떄 가중치는 랜덤으로 부여 ex)0.3
# 예측: 모델에 입력 데이터가 들어가면 입력값들이 가중치와 곱해지고 출력에 나옴
# 오차 계산: 모델이 계산한 예측값과 실제(정답)과 차이 계산, 정답 100 예측 20 => 80
# 가중치 업데이트: 경사하강법을 이용하여 오차가 최소화되는 방향으로 가중치 조절(가중치가 낮아지는 방향)
# 가중치 업데이트 반복

# 경사하강법
# 경사하강법은 가중치 조절하여 손실함수(loss function)를 최소화하는 방법
# 경사하강법 작동원리
# 오차를 최소화하는 목적이기 때문에 실제값과 예측 값의 차이를 통해 정의됨
# mse (mean squared error)를 손실함수 사용

# 기울기 계산 : 경사하강법은 기울기를 계산하고 기울기란 손실함수의 미분값을 의미
# 기울기가 크면 가중치를 많이 조정
# 기울기가 작으면 조정 폭 조금

# 파라미터 업데이트 : 가중치를 계산하고 가중치를 업데이트함 => 이 과정을 반복해서 손실함수가 최소화되도록 함

# 학습률: 가중치를 얼마나 조정할지를 결정하는 학습률 파라미터 , 너무 크면 학습이 불안정, 낮으면 속도 느림



# 1. 가중치 : 학습 과정이 반복되면서 가중치가 계속 조정
# 2. 학습률: learning rate 얼마나 큰 폭으로 가중치를 업데이트할지
# 3. MSE 평균제곱오차 : MSE가 적다 => 가중치가 그럴싸함
# 4. 기울기는 손실함수가 가중치에 대해 얼마나 민감한지(큰지)


# 모델의 훈련에서 예폭과 배치
# 모델 훈련은 데이터를 통해 패턴 학습하는 과정 , 데이터셋과 레이블(정답치)를 가지고 예측하고
# 그 예측이 얼마나 정확한지 손실함수 계산 후 가중치 업데이트함

# 에포크 epoch
# 에폭은 모델이 학습 데이터를 한 번 모두 학습하는 단위 의미
# 좋은 학습하려면 여러번 반복 예복이 필요함
# 너무 많으면 성능 저하


# 배치
# 훈련 데이터셋이 크면 모든 데이터를 한번에 담기 어려움
# 데이터를 작은 덩어리인 배치로 나누어 처리

# 미니 배치
# 데이터를 여러개 작은 배치로 나누어 각 배치마다 가중치 업데이트 과정을 함

# 배치사이즈
# 한번에 학습할 데이터 샘플 수 설정 값


# 모델 평가 예측
# 테스트 데이터셋(test data 0.2)
# 모델 훈련동안 훈련(Train)데이터에 대해서만 학습함
# 훈련데이터 대한 정확도 높게 훈련이 되도록 테스트데이터에서 정확도가 낮으면 과적합의심 해봐야함
# 테스트데이터는 모델 훈련시 사용하지 않고 모델이 얼마나 형변화 되었는 지

# 성능평가 지표 metrics
# 정확도: 모델이 정확히 예측한 샘플의 비율
# 정확도 = 예측이 맞은 수/전체수
# 손실함수 : 모델이 예측한 결과 실제 정답과 차이 측정하는 함수


# 오버피팅 문제 해결방법: 1. 데이터 전처리 2. 하이퍼파라미터 튜닝 , 3. 데이터 샘플 증가
# 드롭 아웃 dropout : 네트워크(신경망) 일부 뉴런을 임의로 제외 시키는 방법


# import tensorflow as tf
# from tensorflow.keras import layers, models           # 신경망 구성 요소 import
# from tensorflow.keras.datasets import mnist           # MNIST 데이터셋 import
# import matplotlib.pyplot as plt                        # 이미지 시각화를 위한 라이브러리
#
#
# # MNIST 데이터셋 로드 (손글씨 숫자 이미지 데이터)
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#
# # 이미지 데이터 정규화: 픽셀 값(0~255)을 0~1 범위로 변환 (성능 향상 효과)
# X_train, X_test = X_train / 255.0, X_test / 255.0
#
# # 신경망 모델 생성 (Sequential: 순차적으로 레이어를 쌓는 방식)
# model = models.Sequential([
#     layers.Flatten(input_shape=(28, 28)),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')
# ])
#
# # 모델 컴파일 (학습 방식 설정)
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )
#
# # 모델 학습 실행 (훈련 데이터로 5번 반복 학습)
# model.fit(X_train, Y_train, epochs=5)
#
# # 테스트 데이터로 모델 성능 평가
# test_loss, test_acc = model.evaluate(X_test, Y_test)
# print("정확도:", test_acc)
#
# # 테스트 데이터 중 첫 번째 이미지에 대한 예측 수행
# y_pred = model.predict(X_test)
# print("첫 번째 이미지 예측값:", y_pred[0].argmax())   # 가장 높은 확률의 클래스 출력
#
# # 첫 번째 테스트 이미지 시각화
# plt.imshow(X_test[0], cmap=plt.cm.binary)              # 흑백 이미지로 표시
# # plt.title(f"예측: {y_pred[0].argmax()}, 실제: {Y_test[0]}")  # 예측값 vs 실제값
# # plt.axis('off')                                        # 축 제거
# plt.show()

# # mnist : 숫자에 대한 손글씨 데이터 셋
# (X_train, Y_train),(X_test,Y_test) = mnist.load_data()
# # xtrain  훈련 데이터 이미지
# #
#
# #y테슽트 : 테스트 이미지 정답지
# X_train,X_test = X_train/255.0,X_test/255.0
# # 이미지 데이터는 0~255 범위인데 정규화로 0~1로 바꿔줄 것
#
# model = models.Sequential([
#     layers.Flatten(input_shapa=(28,28)),
#     layers.Danse(128,activation='relu'),
#     layers.droput(0.2),
#     layers.Danse(10,activation='softmax')
# ])
# model.complie(optparse='adam',lose='sparse_categorical_crossebtropy',metrics=['accuracy'])
# model.fit(X_train,Y_train,epochs=5)
# test_loss,test_acc=model.eveluate(X_test,Y_test)
# print(test_acc,"정확도")
# y_pred = model.predict(X_test)
# print(y_pred[0].argmax())
# plt.imshow(X_test[0],cmap=plt.cm.binary)
# plt.show()