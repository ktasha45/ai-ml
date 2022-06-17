import tensorflow as tf

class RoIPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_size):
        """한 이미지 데이터로 부터 나온 roi들을 vgg model로 부터 추출된 feature map 상에 projection 시키고 
           각각의 roi들을 동일한 크기의 feature vector로 만들어 주는 layer

        Args:
            pool_size (int): fully connected layer에 들어가기 전 사이즈를 맞춰주기 위한 feature vector의 grid 크기 
        """

        tf.keras
        super(RoIPoolingLayer, self).__init__(name='RoI_Pooling_Layer')
        self.pool_size = pool_size # VGG16에서는 7*7
        
    def build(self, input_shape): # input shape로 (1,14,14,512)와 같이 받으니까 3번 원소 자리의 값인 512를 가져간다. 
        self.nb_channels = input_shape[3] # 채널 조정
        # 맨처음 입력받을 때 채널 숫자를 받는다. 풀링이라 채널 개수를 유지해야하기 때문

    def compute_output_shape(self, input_shape): # If the layer has not been built, this method will call build on the layer. 
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, image, RoI_inImage):
        """Shared Feature map과 roi리스트를 인자로 받아 roi pooling 연산을 진행한다 

        Args:
            image (ndarray): vgg 모델로 부터 추출된 feature map 
            RoI_inImage (list): 하나의 이미지로 부터 추출된 roi 리스트 (최대 2000개)

        Returns:
            tensor 리스트: 사이즈가 동일하게 조절된 feature map ex) pool_size = 7, vgg 모델인 경우 -> (1, roi 개수, 7, 7, 512) 여기서 512는 self.nb_channels과 동일하다 
        """

        RoiPooling_outputs_List = [] # RoI 부근을 잘라낸 뒤 7*7로 만들어낸 것들을 여기에 모은다. 그러면 (n,1,7,7,512)가 되겠지

        for i in range(0, len(RoI_inImage)): # 이미지 당 RoI 갯수만큼 for문 반복 -> RoI 갯수만큼 특성맵 얻으려고
            # 224 -> 14로 16배 줄어들었으니 이에 맞춰 RoI도 줄인다. 
            # 기존 RoI 좌표: (x,y,w,h) / 논문에서 언급한 좌표: (r,c,w,h) -> 기존 x,y 좌표가 논문에서 언급한 좌표 (r,c)(왼쪽 위 꼭지점 좌표)와 동일 
            
            r = RoI_inImage[i][0] 
            c = RoI_inImage[i][1] 
            w = RoI_inImage[i][2]
            h = RoI_inImage[i][3]
            
            # 1/16배로 만들기
            r = round(r / 16)
            c = round(c / 16)
            w = round(w / 16)
            h = round(h / 16)
            
            image_inRoI = image[:, c:c+h, r:r+w, :] # RoI에 해당되는 부분을 추출한다.
            image_resize = tf.image.resize(image_inRoI, (self.pool_size, self.pool_size)) # 7*7로 resize
            RoiPooling_outputs_List.append(image_resize)

        # RoiPooling_outputs_List는 (1,7,7,512) 텐서들로 이루어진 리스트다
        final_Pooling_output = tf.concat(RoiPooling_outputs_List, axis=0) # [resize_RoI, resize_RoI, resize_RoI]...리스트를 하나의 텐서로 통합. 밑에 붙히는 방식으로 쫘라락 붙힌다

        final_Pooling_output = tf.reshape(final_Pooling_output, (1, len(RoI_inImage), self.pool_size, self.pool_size, self.nb_channels)) # 통합한걸 (1,RoI 개수,7,7,512)로 reshape
        return final_Pooling_output # (1,RoI 개수,7,7,512) 텐서를 반환
    
    def get_config(self): # 구성요소 반환
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoIPoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))