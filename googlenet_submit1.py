from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import numpy as np

dataset = r'/home/ubuntu/cs249_final_project/train'

from tflearn.data_utils import build_hdf5_image_dataset
#build_hdf5_image_dataset(dataset, image_shape=(300, 300), mode='folder', output_path='dataset.h5', categorical_labels=True, normalize=True)

import h5py
h5f = h5py.File('dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']

#img_aug = tflearn.ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_90degrees_rotation (rotations=[0, 2])

network = input_data(shape=[None, 300, 300, 3])
conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
pool1_3_3 = max_pool_2d(conv1_7_7, 2,strides=2)
pool1_3_3 = local_response_normalization(pool1_3_3)
conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
conv2_3_3 = local_response_normalization(conv2_3_3)
pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

# merge the inception_3a__
inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

#merge the inception_3b_*
inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
pool5_7_7 = dropout(pool5_7_7, 0.4)
loss = fully_connected(pool5_7_7, 3,activation='softmax')
network = regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
model = tflearn.DNN(network, checkpoint_path='model_googlenet',
                    max_checkpoints=1, tensorboard_verbose=2)
#model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
#          show_metric=True, batch_size=64, snapshot_step=200,
#          snapshot_epoch=False, run_id='googlenet_cs249_aug')
#model.save('googlenet_aug')
model.load('model_googlenet-7300')

test_path1 = '/home/ubuntu/cs249_final_project/image_files/test' 
test_path2 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_1'
test_path3 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_2'
test_path4 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_3'
test_path5 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_4'
test_path6 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_5'
test_path7 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_6'
test_path8 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_7'
test_path9 = '/home/ubuntu/cs249_final_project/image_files/test_stg2_8'


import os
f = open('test1.txt', 'w')
for filename in os.listdir(test_path1):
    f.write(os.path.join(test_path1, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test2.txt', 'w')
for filename in os.listdir(test_path2):
    f.write(os.path.join(test_path2, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test3.txt', 'w')
for filename in os.listdir(test_path3):
    f.write(os.path.join(test_path3, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test4.txt', 'w')
for filename in os.listdir(test_path4):
    f.write(os.path.join(test_path4, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test5.txt', 'w')
for filename in os.listdir(test_path5):
    f.write(os.path.join(test_path5, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test6.txt', 'w')
for filename in os.listdir(test_path6):
    f.write(os.path.join(test_path6, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test7.txt', 'w')
for filename in os.listdir(test_path7):
    f.write(os.path.join(test_path7, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test8.txt', 'w')
for filename in os.listdir(test_path8):
    f.write(os.path.join(test_path8, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

f = open('test9.txt', 'w')
for filename in os.listdir(test_path9):
    f.write(os.path.join(test_path9, filename) + ' ' + os.path.splitext(filename)[0] + '\n')
f.close()

#build_hdf5_image_dataset('test1.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset1.h5')
h5f1 = h5py.File('testset1.h5', 'r')

#build_hdf5_image_dataset('test2.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset2.h5')
h5f2 = h5py.File('testset2.h5', 'r')

#build_hdf5_image_dataset('test3.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset3.h5')
h5f3 = h5py.File('testset3.h5', 'r')

#build_hdf5_image_dataset('test4.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset4.h5')
h5f4 = h5py.File('testset4.h5', 'r')

#build_hdf5_image_dataset('test5.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset5.h5')
h5f5 = h5py.File('testset5.h5', 'r')

#build_hdf5_image_dataset('test6.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset6.h5')
h5f6 = h5py.File('testset6.h5', 'r')

#build_hdf5_image_dataset('test7.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset7.h5')
h5f7 = h5py.File('testset7.h5', 'r')

#build_hdf5_image_dataset('test8.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset8.h5')
h5f8 = h5py.File('testset8.h5', 'r')

#build_hdf5_image_dataset('test9.txt', image_shape=(300, 300), mode='file', categorical_labels=False, output_path='testset9.h5')
h5f9 = h5py.File('testset9.h5', 'r')

X = h5f1['X']
predict1 = model.predict(X)
X = h5f2['X']
predict2 = model.predict(X)
X = h5f3['X']
predict3 = model.predict(X)
X = h5f4['X']
predict4 = model.predict(X)
X = h5f5['X']
predict5 = model.predict(X)
X = h5f6['X']
predict6 = model.predict(X)
X = h5f7['X']
predict7 = model.predict(X)
X = h5f8['X']
predict8 = model.predict(X)
X = h5f9['X']
predict9 = model.predict(X)


Y1 = h5f1['Y']
Y2 = h5f2['Y']
Y3 = h5f3['Y']
Y4 = h5f4['Y']
Y5 = h5f5['Y']
Y6 = h5f6['Y']
Y7 = h5f7['Y']
Y8 = h5f8['Y']
Y9 = h5f9['Y']

import csv
with open('submit.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['image_name','Type_1','Type_2','Type_3'])
    for i in range(len(predict1)):
        filename = str(int(Y1[i])) + '.jpg'
        writer.writerow([filename] + predict1[i])
    for i in range(len(predict2)):
        filename = str(int(Y2[i])) + '.jpg'
        writer.writerow([filename] + predict2[i])
    for i in range(len(predict3)):
        filename = str(int(Y3[i])) + '.jpg'
        writer.writerow([filename] + predict3[i])
    for i in range(len(predict4)):
        filename = str(int(Y4[i])) + '.jpg'
        writer.writerow([filename] + predict4[i])
    for i in range(len(predict5)):
        filename = str(int(Y5[i])) + '.jpg'
        writer.writerow([filename] + predict5[i])
    for i in range(len(predict6)):
        filename = str(int(Y6[i])) + '.jpg'
        writer.writerow([filename] + predict6[i])
    for i in range(len(predict7)):
        filename = str(int(Y7[i])) + '.jpg'
        writer.writerow([filename] + predict7[i])
    for i in range(len(predict8)):
        filename = str(int(Y8[i])) + '.jpg'
        writer.writerow([filename] + predict8[i])
    for i in range(len(predict9)):
        filename = str(int(Y9[i])) + '.jpg'
        writer.writerow([filename] + predict9[i])