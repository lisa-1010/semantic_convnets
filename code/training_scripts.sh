#!/bin/bash
# Script to train the pyramid model from previous checkpoint of the same model
python pipeline.py -t train -m "pyramid_cifar100" -c "pyramid_cifar100"