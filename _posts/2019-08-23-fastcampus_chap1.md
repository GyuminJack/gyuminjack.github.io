---
layout: post
title: 패스트캠퍼스 CNN중심 tensorflow 2.0
category : deep learning
---
- chapter1 (CNN중심)

- NN은 logit을 통해 loss를 계산하고 반복적으로 optimize
- 용어
	- Model 
	- Layer : 깊이 쌓으면 representation이 좋아지지만, overfit 가능성 올라감
	- Convolution : 합성곱, 일종의 필터역할을 하면서 특성을 뽑아냄
	- Pooling : 데이터의 압축을 위해 사용됨, 사이즈를 줄이는 필터를 적용
		- max_pooling 등의 다양한 방식이 있음
- CNN 모델 구조
	- Feature Extraction : 특징을 추출하는 convolution층
	- Classfication : convolution 이후에 Fully Connected 하는 층
	- Feature Extraction은 Convolution, pooling과정을 거치게 됨
		- 필터역할을 해서 데이터의 중요한 부분만 추출함
		- max_pooling은 그리드 중에 가장 큰 값을 가져오면서 압축을 진행
		- activation function은 각각의 값에 대해 적용되어 비선형적으로 가져오게됨
	- CNN의 경우 레이어가 그다지 어렵지는 않고, 깊은 신경망에도 효율적인 CNN이 만들어지는 구조가 좋은 구조체임

- chapter 2 (tensorflow, pytorch)
	- tf : 1.x 보다 훨씬 편해짐 / TensorBoard 가 중요할 듯..
	- pytorch : pythonic 함이 있음.

- tensorlfow 2.0 (CNN/keras)
	- 상수 생성
		- tf.constant : 고정길이의 tensor, 그런데 np.array를 훨씬 많이 쓰게됨.
		- tensor를 만들때 dtype을 생각하고 진행하는게 중요, 디버깅시에 어려울수도 있음
		- tf.cast : tensor의 타입을 변경 (float->int..)
	- 난수 생성
		- tf.random.normal([shape]) : shape에 맞는 난수 생성
		- tf.random.uniform([shape]) : shape에 맞는 난수 생성
	- mnist 데이터 불러오기
		- mnist = tf.keras.datasets.mnist
		- (train_x, train_y), (test_x, test_y) = mnist.load_data() (Convention임)
		- 이미지의 경우 shape이 [total_N, 가로, 세로, 채널]
		- 채널은 이미지의 색 구성, 1은 gray scale, 3은 RGB
	- 차원 늘리기
		- np.expand_dims(x, -1) : -1번째에 차원을 늘림(맨뒤에 1 차원을 넣어줌, 28\*28만 있을때 사용)
		- tf.expand_dims(x, -1) / x[...,tf.newaxis] 두개는 동일, 후자가 깔끔함.
		- tf.reshape 도 가능함
		- matplotlib은 무조건 2차원이므로 시각화를 위해선 마지막 dim을 삭제해야함.
	- 원핫 인코딩
		- tf.keras.utils.to_categorical(label, num_classes)
	- Layer
		- layer 쌓기 전에 무조건 shape를 확인 해야됨.
		- input은 무조건 [batch_size, heigth, width, channel]로 들어가야됨
		- Feature Extraction 과정과 Classification 과정을 거치는 게 일반적
		- Feature Extraciton
			- tf.keras.layers.Conv2d(filter, kernel_size, strides, padding, activation)
				- filter : convolution을 걸치고 나온 데이터가 몇개의 channel을 가지게 되는지
				- strides : 필터가 건너가는 크기
				- padding : zero padding의 경우 크기가 그대로임
				- acivation : 어떤 actvation function을 쓸건지, None이 기본값.
			- Convolution 이후 나오는 이미지 확인 (plt.imshow(output[0,:,:,0]))
				- layer = tf.keras.layers.Conv2d(3,3,1,"SAME")
				- output = layer(input)
				- layer_weight = layer.get_weights() -> 2개의 인자, 앞에 것은 weight 뒤에것은 bias
				- Activation : Relu 적용 (tf.keras.layers.ReLU())
				- max_pooling : tf.keras.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='SAME')
					- 풀링은 일반적으로 2,2가 쓰임
		- Classification
			- Fully Connected
	- Keras version
		- Loss
			- Binary / Categorical Cross Entropy
			- sparse_categorical_crossentropy(원핫인코딩 안해도 쓸수있음) / categorical_crossentropy(원핫해야됨)
		- metrics
			- tf.keras.metrics.Accuracy() ...
		- optimizer
			- tf.keras.optimizer.Adam()
		- model compiling
			- model.compile(optimzer, loss, [metrics])
		- insert Data setting
			- 1. 입력값 차원수가 4 인지 채널 값 확인 / train_x[...,tf.newaxis]
			- 2. rescale : min-max, etc..
			- 3. epoch, batch_size
		- training
			- model.fit(train_x, train_y, batch_size, epochs, shuffle)
	- Expert version
		- tf.data로 데이터 로딩
			- train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
			- train_ds = train_ds.shuffle(1000) 1000정도가 적당
			- train_ds = train_ds.batch(32)
			- test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
		- 데이터 불러오기 (제너레이터)
			- image, label = train_ds.take(2)
			- train_ds는 무조건 batch size 설정한 만큼만 가져옴 / 위의 예제는 총 64개를 가져옴
		- model.compile(optimizer, loss)
		- model.fit(train_ds, epochs)
		- loss를 직접 구현 하는 방식
			~~~python
			# training phase
			@tf.fuction
			def train_step(image, labels):
				with tf.GradientTape() as tape:
					prediction = model(images)
					loss = loss_object(labels, predictions)
				gradient = tape.gradient(loss, model.trainable_variables)
				optimizer.apply_gradient(zip(gradients, model.trainable_variables))
				train_loss(loss)
				train_accuracy(labels, prediction)			# training phase
			# test phase
			@tf.fuction
			def test_step(image, labels):
				prediction = model(images)
				loss = loss_object(labels, predictions)
				test_loss(t_loss)
				train_accuracy(labels, prediction)
			for epoch in range(2):
				for image, label in train_ds:
					train_step(image, label)
				for test_image, test_labels in test_ds:
					test_step(test_images, test_labels)
				template = 'Epoch {}, Loss : {}, ...'
				print(template.format(train_loss.result(), train_accuracy.result()))
			~~~











