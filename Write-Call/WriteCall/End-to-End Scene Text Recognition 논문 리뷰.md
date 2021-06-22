<p align="center">
  <img src="https://user-images.githubusercontent.com/59414764/112703159-46b13c80-8ed9-11eb-9b81-c8e34cd8bde1.png"/>
</p>
<br/>

# Abstract
Natural Image상에서 글자를 찾는 것은 서류를 스캔하는 것보다 어렵다. 본 논문에서는 두가지 시스템을 구축하고 평가하여 기존의 word recognition을 보완한다.
첫번째로, 1단계 text detection, 2단계 OCR engine로 이루어진 pipeline 방법이다. 두번쨰로는 generic 객체 탐색이다. 일반적으로 text recognition은 영역을 특정하는 방법을 이용하지만, 본 논문에서는 generic(유전적) 컴퓨터 비전 방법이 적합함을 설명해준다. 이러한 방법은 빠르다는 이점을 가지고 있다.

# 1. Introduction
 어떠한 형태로 정규화 되지 않은 이미지에서 단어를 읽는 건 정말 힘든 일이다. 서류 스캔은 예전부터 OCR 기술을 사용해온 반면에, 모바일 카메라 장치가 많이 쓰이면서, 이미지에서 scene text를 얻는 과정이 점점 보편화 되고 있다. 텍스트는 우리 환경에서 흔히 볼 수 있기 때문에, scene text를 얻는 문제는 점점 중요해지고 있다. 예를 들어, scene text를 읽는 것은 외부 에서 자율 주행의 네비게이션 기능을 할 때 중요한 역할을 하고, 실내 환경에서 시각 장애인의 네비게이션 역할도 수행한다.
 명백하게 중요함에도 불구하고, scene text 문제는 컴퓨터 비전 커뮤니티 상에서 많이 주목받지 못했다. ICDAR Robust Reading challenge는 처음으로 scene text 인식하는 것에 중점을 둔 공공 데이터셋이다. 이 단체의 관계자는 4가지의 문제를 정의했다. (1) 문자를 분류해내는 것, (2) 전체 이미지에서 텍스트를 찾는 것, (3) 단어를 인식하는 것, (4) 전체 이미지에서 텍스트를 인식하는 것. 본 논문은 찾을 단어 목록이 주어졌을 때, scene text 문제를 해결할 방법에 대해 다룰 것이다. 만약 시각 장애인이 식료품점을 찾고 있다면, 연관 단어를 찾아낼 것이다. 객체 인식 기반의 text detection 기술은 OCR과 비교했을 때, 뛰어난 성능을 보여주었다.
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/59414764/113071516-d91a4e80-91ff-11eb-959e-a3531af15a50.png"/>
 </p>

# 2. Overview of Full Image Word Recognition
## 2.1. Character Detection 
 첫번째 단계는 이미지에서 문자의 위치를 알아내는 것이다. 본 논문에서는 sliding window classification을 통한 multi-scale character detection을 수행하였다. 발견해야할 문자의 카테고리가 넓기 때문에, 분류기를 잘 골라야한다. 이 논문에서는 Random Ferns(참조: https://darkpgmr.tistory.com/90, http://randomferns.blogspot.com/2014/03/random-ferns.html) 을 사용하였다.
