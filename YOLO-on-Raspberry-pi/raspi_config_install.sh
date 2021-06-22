sudo apt update
sudo apt -y upgrade
git clone https://github.com/EmilGus/install_raspi-config
sudo sh install.sh
vcgencmd get_camera
raspistill -o test.jpg
