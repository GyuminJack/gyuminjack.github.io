---
layout: page
title: It's Me
---

##### 함께 하는 것과 함께 나누는 것을 좋아하는 데이터 분석가 입니다.
- mail : sld9849@gmail.com

##### 프로젝트 소개
- J 기관 트윗 분석 고도화
    - 기존에 기관에서 소유하고 있는 소셜 분석 플랫폼 고도화
    - 신규 분석 개발의 경우 word2vec을 통한 연관 단어 추출, LDA를 통한 재난 내 단어 분포 파악 두가지 구현
    - 신규 형태소 분석기 비교 후 명사추출기를 통해 연산량 최소화
    - word2vec은 네트워크 그래프로 시각화 / LDA의 경우 개별 단어들의 분포를 시각화, 각 단어를 가진 트윗의 추이를 함께 보여줌

    개발환경
    - 백엔드 : python

    개발물
    사진 [시계열 그래프 사진]
    사진 [word2vec 사진]

- 특허 등록
    - 이동통신데이터 처리시스템의 이상로그 발생을 진단하는 방법 및 그 시스템 (Method for diagnosing anomaly log of mobile commmunication data processing system and system thereof)

- K 기관 IoT 보안 연구 과제
- 과제 기간 : 총 4년(2018~)
    - 가정 내 IoT의 기기들의 트래픽을 구분하고 각각의 신경망 모델을 통해 이상 트래픽 탐지
    - 이상트래픽 탐지 장비의 경우 Home환경의 AP로 설정하고 스펙은 라즈베리파이급으로 설정.
    - 딥러닝의 경우 빠른 학습이 필수적이었고, 탐지 장비의 AP,DHCP 등 기능 관리와 AD결과에 대한 UI도 필요했음.
    - 탐지 장비의 경우 클라우드와 연동되어야 하기 때문에 통신에 대한 REST API 개발

    개발 환경
    - 코드 관리 : github
    - 패킷 수집 / 정제 : tcpdump, tshark
    - 패킷 전처리 : python(pandas, numpy) + multiprocessing
    - 딥러닝 모듈 : python(tensorflow)
    - 관리 웹 : php+lighttpd, python
    
    AD 모델
    - NN-Training : SVDD 알고리즘을 사용했고 두가지의 과정을 거침
        - phase 1 : Autoencoder를 통한 데이터 압축 후 SVDD loss에 사용될 center vector생성
        - phase 2 : MLP + SVDD loss를 통한 학습
            - 학습의 경우 early_stopping 과 pruning을 통해 경량화
            - SVDD의 경우 Autoencoder를 통한 AD의 문제점인 thresholds 문제에서 자유로움
    - NN-Test : 기존 모델 구조를 API서버를 통해 pre loading
        - inferrence 시에는 POST방식으로 최대한 빠르게 inferrence
        - flask서버의 경우 단일 쓰레드를 사용해서 느려지기 때문에 gunicorn으로 wrap-up 해 멀티쓰레딩 구현
    
    관리 UI
    - 처음부터 끝가지 구현하는 것보다 오픈소스를 포팅하는 것이 효과적이라고 판단.
    - github에 공개된 rasp-ap ui를 포팅
    - 포팅후 한글화, AD관리 화면, 클라우드 연동 API등 진행.

- K 기관 5G 네크워크 보안 연구 과제
- 과제 기간 : 총 4년(2019.08~)
    - 5G 네트워크 트래픽에 대한 이상탐지
    - 시그니처 기반의 탐지를 보완할 수 있는 머신러닝 탐지 알고리즘 개발이 목적
    - 5G 프로토콜의 이해를 통한 피처 추출 과정 진행