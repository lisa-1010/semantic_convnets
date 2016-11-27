# semantic_convnets

# Instructions to create an AWS P2/G2 Instance

Tensorflow AMI id: ami-133ae673

(1) Spin up spot instance for specified time period.

### Note the ip address here: 35.163.93.248 ###

(2) SSH into the instance (don't forget to add security groups to allow ssh)

ssh -i (path to pem file) ubuntu@(ip address)

(3) Install the following:

sudo pip install --upgrade pip 
sudo pip install h5py
sudo pip install sklearn
sudo pip install git+https://github.com/tflearn/tflearn.git

(4) Clone repo and copy over data files

git clone https://github.com/lisa-1010/semantic_convnets

(5) Copy over data

scp -r -i {path to pem file} data ubuntu@{ip address}:~/semantic_convnets

(6) That's it! Run python pipeline.py to train the models. You can also run tensorboard, but make sure you setup the network security groups. Also, don't forget to terminate the instance / copy back and forth the data. 



