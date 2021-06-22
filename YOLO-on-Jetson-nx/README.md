# YOLO
## 개발 환경
Jetson NX - Jetpack 4.4.1 환경에서 진행하였습니다 (about jetpack : "https://developer.nvidia.com/embedded/jetpack")
Jetpack 4.4.1 환경:
- Linux Kernel 4.9
- Ubuntu 18.04
- TensorRT: CUDA와 NVIDIA의 병렬 처리 작업을 위한 딥러닝 프레임 워크
- cuDNN: GPU 가속 응용앱을 위한 C/C++ 개발 환경
- OpenCV 4.1.1
- GCC 7.3.1 for 64 bit BSP and Kernel
- python 3.6.9, python 2.7.17

## darknet 설치
: git clone https://github.com/AlexeyAB/darknet.git

darknet 폴더의 Makefile 수정

GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=0
AVX=0
OPENMP=0
LIBSO=0

위의 내용을 아래와 같이 수정

GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=0
LIBSO=1

: make

## 실행방법
: python yolo_test_DK.py

![saved_image](https://user-images.githubusercontent.com/59414764/101979859-09b39800-3ca4-11eb-95b4-6bcac444ca87.png)
