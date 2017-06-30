'''
Title           :Predictions_2.py
Description     :Makes predictions using caffe_model_2 and generates a csv submission file.
Author          :Nitin Shukla
Date Created    :20170624
'''

import os
import glob
import sys
sys.path.append('/home/ubuntu/caffe/python')

import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_cpu()

#Spatial dimension
IMG_WIDTH = 227
IMG_HEIGHT = 227

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
net = caffe.Net('/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/caffe_models/caffe_model_2/deploy_2.prototxt',
                '/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/caffe_models/caffe_model_2_iter_3000.caffemodel',
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
test_ids = []
preds = []
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    print pred_probas.argmax()
    print '-------'

'''
Making submission file
'''
with open("/home/ubuntu/DeepLearning_crop_classification/Crop_Classification/caffe_models/caffe_model_2/submission_model_2.csv","w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(preds[i])+"\n")
f.close()
