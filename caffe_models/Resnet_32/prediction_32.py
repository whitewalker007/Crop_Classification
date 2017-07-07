'''
Title           :Prediction_32.py
Description     :Makes leaves based predictions and generates a csv file.
Author          :Nitin Shukla
Date Created    :20170707
'''

import os
import glob
import sys
sys.path.append('/home/ubuntu/src/caffe')

import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()

#Spatial dimension
IMG_WIDTH = 224
IMG_HEIGHT = 224

'''
Processing
'''

def transform_img(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/ubuntu/DeepLearning_crop_classification/input_2/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/Resnet_32/deploy_32.prototxt',
                '/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/Resnet_32/Resnet_32_iter_5000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/ubuntu/DeepLearning_crop_classification/input_2/test/*jpg")]

#Making predictions
with open("/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/Resnet_32/Resnet_crop_32.csv","w") as f:
    f.write("Id,LEAVES,PLANT,PLOT,LABEL\n")
    for img_path in test_img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        pred_probas = out['prob']

        test_ids = img_path.split('/')[-1][:-4]
        preds = pred_probas.argmax()

        print img_path
        print pred_probas
        print '-------'


        f.write(str(test_ids)+","+str(pred_probas[0,0])+","+str(pred_probas[0,1])+","+str(pred_probas[0,2])+","+str(preds)+"\n")
    f.close()
