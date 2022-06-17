import random

import numpy as np
import tensorflow as tf
from RoIPoolingLayer import RoIPoolingLayer

class FastRCNN(tf.keras.Model):
    def __init__(self, SharedConvNet):
        super(FastRCNN, self).__init__(name='fastrcnn')

        self.Optimizers = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

        # 공용 컨볼루전 레이어들 추가
        self.conv1_1 = SharedConvNet.layers[0]
        self.conv1_2 = SharedConvNet.layers[1]
        self.pooling_1 = SharedConvNet.layers[2]

        self.conv2_1 = SharedConvNet.layers[3]
        self.conv2_2 = SharedConvNet.layers[4]
        self.pooling_2 = SharedConvNet.layers[5]

        self.conv3_1 = SharedConvNet.layers[6]
        self.conv3_2 = SharedConvNet.layers[7]
        self.conv3_3 = SharedConvNet.layers[8]
        self.pooling_3 = SharedConvNet.layers[9]

        self.conv4_1 = SharedConvNet.layers[10]
        self.conv4_2 = SharedConvNet.layers[11]
        self.conv4_3 = SharedConvNet.layers[12]
        self.pooling_4 = SharedConvNet.layers[13]

        self.conv5_1 = SharedConvNet.layers[14]
        self.conv5_2 = SharedConvNet.layers[15]
        self.conv5_3 = SharedConvNet.layers[16]

        self.RoI_Pooling_Layer = RoIPoolingLayer(7) # Pooling 이후 크기를 7*7*512로 만든다. -> (1,num_roi,7,7,512)
        self.Fully_Connected = tf.keras.layers.Dense(4096, activation='relu')  # num_roi*[1, 7*7*512] 텐서를 받는다 -> num_roi*[7*7*512, 4096]
        self.Classify_layer = tf.keras.layers.Dense(5, activation='softmax', name = "output_1") # num_roi*[4096,5]
        self.Reg_layer = tf.keras.layers.Dense(4, activation= 'relu', name = "output_2") # num_roi*[4096, 4]
    
    def call(self, Image, RoI_list): 
        """순전파
           입력 데이터로 이미지 1개, 이미지에 해당하는 roi리스트를 받고
           클래스를 예측하는 feature vector와 예측한 물체에 해당하는 박스 feature vector를 얻는다 

        Args:
            Image (ndarray): 이미지 데이터
            RoI_list (ndarray): 이미지 데이터에 대응되는 RoI들

        Returns:
            cls_output, reg_output (tensor, tensor): roi별 feature vector
        """

        output = self.conv1_1(Image)
        output = self.conv1_2(output)
        output = self.pooling_1(output)

        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.pooling_2(output)

        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.conv3_3(output)
        output = self.pooling_3(output)

        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.conv4_3(output)
        output = self.pooling_4(output)

        output = self.conv5_1(output)
        output = self.conv5_2(output)
        shared_output = self.conv5_3(output)
        
        # feature map과 RoI를 인자로 받고 (1, RoI 개수, 7,7,512) shape의 feature vector를 얻는다 
        pooling_output = self.RoI_Pooling_Layer(shared_output, RoI_list) 
        # roi별 flatten vector 구성 
        rois_flatten = tf.reshape(pooling_output, (1, 64, 7*7*512))

        Fully_Connected_output = self.Fully_Connected(rois_flatten) 
        # 객체 분류 레이어, 박스 회귀 레이어
        cls_output = self.Classify_layer(Fully_Connected_output) 
        reg_output = self.Reg_layer(Fully_Connected_output)
        
        return cls_output, reg_output 
