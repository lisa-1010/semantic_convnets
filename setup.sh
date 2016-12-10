#!/bin/bash

# Usage: source setup.sh
#        If you pass an argument, then you SSH into the instance instead of running the setup scripts.

IPADDR="35.161.86.147"

if [ $# -eq 1 ]; then
    ssh -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR
    exit 1
fi

ssh -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR <<-'ENDSSH'
    #commands to run on remote host
    cd ~/
    sudo pip install --upgrade pip
    sudo pip install h5py
    sudo pip install sklearn
    sudo pip install git+https://github.com/tflearn/tflearn.git
    git clone https://github.com/lisa-1010/semantic_convnets
    sudo pip install -r semantic_convnets/requirements.txt
ENDSSH

scp -r -i ~/.ssh/MyndbookEBKeyPair.pem data ubuntu@$IPADDR:~/semantic_convnets




