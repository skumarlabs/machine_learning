#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 03:23:09 2017

@author: suri
"""

class BatchGenerator():
    def __init__(self):
        pass
    def generate(self, tfrecords_files):
        ''' tfrecords_files is the list of TFRecords files having examples. in our case only one'''
        with tf.Session() as sess:
            feature = {
                        'image' : tf.FixedLenFeature([], tf.string),
                        'label' : tf.FixedLenFeature([], tf.int64)
                    }
            filename_queue = tf.train.string_input_producer([tfrecords_files], num_epochs = 1)
            
            # define a reader and read the next record
            reader = tf.TFRecordReader()
            
            
            _, serialized_example = reader.read(filename_queue)
            
            # decode the record read by reader
            features = tf.parse_single_example(serialized_example, features = feature)
            
            # convert the image data from string back to numbers
            image = tf.decode_raw(features['image'], tf.float32)
            
            
            # cast label data into int32
            label = tf.cast(features['label'], tf.int32)
            
            # reshape image data into original shape
            image = tf.reshape(image, [224, 224, 3])
            
            # any preprocessing here
            
            # create batches by randomly shuffling tensors
            images, labels = tf.train.shuffle_batch([image, label],
                                                    batch_size=10,
                                                    capacity=30,
                                                    num_threads=1,
                                                    min_after_dequeue=10)
            
            # initialize all local and global variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            
            # create a co-ordicator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)
            
            for batch_index in range(5):
                img, lbl = sess.run([images, labels])
                
                img = img.astype(np.uint8)
                for j in range(6):
                    plt.subplot(2, 3, j+1)
                    plt.imshow(img[j, ...])
                    plt.title('cat' if lbl[j] == 0 else 'dog')
                plt.show()
                
            # stop thread
            coord.request_stop()
            
            # wait  for thread to stop
            coord.join(threads)
            sess.close()
            
