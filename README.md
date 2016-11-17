# semantic_convnets

# Instructions to create an AWS P2/G2 Instance

Tensorflow AMI id: ami-133ae673

1) Spin up spot instance for specified time period. Note the ip address: 
	ip address: 35.162.225.37
2) SSH into the instance
	ssh -i (path to pem file) ubuntu@(ip address)
3) Install
	pip install tflearn
	sudo pip install h5py
	pip install --upgrade pip 

4) Clone repo and copy over data files
	git clone https://github.com/lisa-1010/semantic_convnets

5) Copy over data
	scp -r -i {path to pem file} data ubuntu@{ip address}:~/semantic_convnets

6) That's it! Run python pipeline.py to train the models.


