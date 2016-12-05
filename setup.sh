#!/bin/bash
IPADDR="35.165.30.83"

ssh -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR <<-'ENDSSH'
    #commands to run on remote host
    cd ~/
    sudo pip install --upgrade pip
    sudo pip install h5py
    sudo pip install sklearn
    sudo pip install git+https://github.com/tflearn/tflearn.git
    git clone https://github.com/lisa-1010/semantic_convnets
ENDSSH

scp -r -i ~/.ssh/MyndbookEBKeyPair.pem data ubuntu@$IPADDR:~/semantic_convnets




