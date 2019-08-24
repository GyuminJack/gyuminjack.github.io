---
layout: post
title: training이 안되는 이유 찾기
category : deep learning
---

training이 잘 안될때 만큼 신경쓰이는게 없다..
- loss가 너무 떨어져서 overfitting (validation set으로 확인해야)
    - overfitting이 underfitting보다 잘 일어 나는 느낌이 있음.
- 아예 loss자체가 큰 값(>50)에 머무는 경우.
    - custom loss를 사용했을때 이런 경우가 발생했음.
    - normalize를 안했을때 이런경우 발생.
- pandas dataframe을 바로 session으로 넣었을때 문제 발생. 


잘 안되는 이유.
- 1. batch size의 크기
    - 가능하면 큰 값으로 시행
- 2. learining rate의 크기
    - 일정 epoch 이후로 계속 그자리에 머문다면 learning rate를 좀 줄여서 training
- 3. 