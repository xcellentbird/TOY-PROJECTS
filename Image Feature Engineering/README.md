# Image Feature Engineering with AutoEncoder
- Tranposed Conv를 이용하여 AutoEncoder 모델을 간단하게 만들어보고, Encoder를 통해 Feature를 얻어냈습니다.

# KMeans Clustering
- 데이터 선정: MNIST Fashion dataset
- 위의 encoder에서 얻은 feature를 이용하여 clustering을 수행하였고, class 기준으로 clustering 결과를 해석하였습니다.

# 실험 결론
다음과 같은 가설을 세울 수 있었습니다.
- 다양한 조건에서의 Image Feature Engineering 결과로, n_clusters를 늘려가며 clustering(random_state=None 설정)을 여러번 수행하였을 때, 같은 패턴('특정 데이터가 항상 같은 군집을 이룬다', '군집화 결과를 시각화하였을 때 군집 특징을 육안으로 유추할 수 있다', 등등)의 군집화가 이루어졌을 경우, 이는 의미있는 cluster 또는 Classification으로 판단할 수 있을 것이다.
- 또한 해당 Cluster를 결정하는 주요 Feature를 선정하였을 때, 이 Feature 또는 Feature set을 의미있는 Feature(set)으로 생각할 수 있다.
- 이와 같은 Image Feature Engineering을 통해 이미지나 해당 class를 해석할 수 있을 것으로 보인다. 또한, 모델이 어떠한 특징을 잡아내는지 알 수 있지 않을까?


>사용 기술 스택: Deep Learning, Machine Learning, Data Analysis, Computer Vision.  
> * *Python, PyTorch, Scikit-Learn, Pandas, Matplotlib, Numpy*
