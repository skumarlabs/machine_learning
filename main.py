#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 23:21:52 2017

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tfrecords_converter import TfConverter
from batch_generator import BatchGenerator

# shuffle_data = True  # shuffle the addresses before saving

cats_dogs_train_path = 'training_set/images/*.jpg'       

conv = TfConverter()
addrs, labels = conv.prepare(cats_dogs_train_path)
train_set, valid_set, test_set = conv.train_test_valid_set(addrs, labels)

#conv.create_records(train_set, 'train_set.tfrecords')
#conv.create_records(test_set, 'test_set.tfrecords')
conv.create_records(valid_set, 'valid_set.tfrecords')
BatchGenerator().generate('valid_set.tfrecords')


        
        
    
    
    
