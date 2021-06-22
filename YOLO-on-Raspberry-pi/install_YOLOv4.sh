cd
sudo apt-get -y install git
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
make -j4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
