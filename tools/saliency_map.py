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
        img_h = img.shape[0]
        img_w = img.shape[1]
        img = cv2.resize(img, (224,224))
        img = np.asarray(img, dtype=np.float32)
        img -= mean_value
        img = img.transpose(2,0,1)
        img = img.reshape((1,) + img.shape)
        net.blobs['data'].reshape(*img.shape)
        net.blobs['data'].data[...] = img

        output = net.forward()
       
        output_prob = output['prob'][0]
        CLASS = output_prob.argmax()

        finalLabelVector = np.zeros((1, 1000))
        finalLabelVector[0, CLASS] = 1

        backwardpassData = net.backward(**{net.outputs[0]: finalLabelVector})
        #print backwardpassData
        delta = backwardpassData['data']
        #print delta.shape

        delta = delta - delta.min()  # Subtract min
        delta = delta / delta.max()  # Normalize by dividing by max
        saliency = np.amax(delta, axis=1)  # Find max across RGB channels


        plt.subplot(1, 2, 1)
        src = cv2.imread(image_root + f)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        src = cv2.resize(src, (224, 224))
        plt.imshow(src)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(saliency[0, :, :], cmap='copper')
        plt.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()
