#!/bin/bash

# Usage: source aws.sh {arg0 arg1}
#        if arg0 == "ssh", then will ssh into the instance
#        if arg0 == "pull", then will pull to current directory from aws instance whatever is at arg1
#        if arg0 == "setup", then will ssh into instance, run setup scripts, and then exit
# For reference: Tensorflow AMI id: ami-133ae673
IPADDR="35.160.50.31"

if [ $# -ne 0 ]; then
    if [ "$1" = "ssh" ]; then
        ssh -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR
    fi
    if [ "$1" = "pull" ]; then
        echo "Pulling to current directory: "
        echo "scp -r -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR:~/$2"
        scp -r -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR:~/$2 .
    fi
    if [ "$1" = "setup" ]; then
        ssh -i ~/.ssh/MyndbookEBKeyPair.pem ubuntu@$IPADDR <<-'ENDSSH'
            #commands to run on remote host
            cd ~/
            sudo pip install --upgrade pip
            sudo pip install h5py
            sudo pip install sklearn
            sudo pip install git+https://github.com/tflearn/tflearn.git
            git clone https://github.com/lisa-1010/semantic_convnets
            sudo pip install matplotlib
            sudo pip install pandas
            sudo pip install seaborn
            sudo apt-get install python-tk
            Y
            sudo pip install -r semantic_convnets/requirements.txt
ENDSSH

        scp -r -i ~/.ssh/MyndbookEBKeyPair.pem data ubuntu@$IPADDR:~/semantic_convnets
    fi
fi



