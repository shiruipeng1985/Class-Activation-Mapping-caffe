import sys
sys.path.append('/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Seg_caffe/caffe-zlh/python')
import caffe
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

caffe.set_mode_gpu()

model_def = '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Class-Activation-Mapping/tools/deploy_googlenetCAM.prototxt'
model_weights = '/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Class-Activation-Mapping/tools/imagenet_googleletCAM_train_iter_120000.caffemodel'

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

mean_value = np.array([104, 117, 123])

image_root='/media/zlh/ccbaea80-0264-47ef-a0ea-6941cc7542f2/Class-Activation-Mapping/image/'
for root,dirs,files in os.walk(image_root):
    for f in files:
        img = cv2.imread(image_root+f)
        img = cv2.resize(img, (224,224))
        img = np.asarray(img, dtype=np.float32)
        img -= mean_value
        img = img.transpose(2,0,1)
        img = img.reshape((1,) + img.shape)
        net.blobs['data'].reshape(*img.shape)
        net.blobs['data'].data[...] = img

        output = net.forward()
        weights = net.params['CAM_fc'][0].data
        conv_img = net.blobs['CAM_conv'].data[0]
        weights = np.array(weights, dtype=np.float)
        conv_img = np.array(conv_img, dtype=np.float)
        heat_map = np.zeros([14, 14], dtype=np.float)

        output_prob = output['prob'][0]
        CLASS = output_prob.argmax()

        for i in range(1024):
            w = weights[CLASS][i]
            heat_map += w * conv_img[i]
        heat_map = cv2.resize(heat_map, (224, 224))

        src = cv2.imread(image_root+f)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (224, 224))
        #heat_map = 100 * heat_map

        #print net.blobs['prob'].data[0][CLASS]
        s = plt.imshow(src)
        s = plt.imshow(heat_map, alpha=0.5, interpolation='nearest')
        plt.show()
