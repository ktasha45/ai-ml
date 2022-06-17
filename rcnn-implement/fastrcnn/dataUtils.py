import random
from tqdm import tqdm
import xmltodict

import cv2 
import os
import numpy as np
import pandas as pd 

class EdgeBoxes():
    def __init__(self, model_path):
        self.model = model_path
        try:
            self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(self.model)
        except:
            print('Invalid model path or the model file is corrupt.')
        
    def __call__(self, image_array, maxBoxes=2000):
        self.im = image_array
        # self.rgb_im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        edges = self.edge_detection.detectEdges(np.float32(self.im)) # /255.0

        orimap = self.edge_detection.computeOrientation(edges)
        edges = self.edge_detection.edgesNms(edges, orimap)

        edge_boxes = cv2.ximgproc.createEdgeBoxes() 
        edge_boxes.setMaxBoxes(maxBoxes)
        return edge_boxes.getBoundingBoxes(edges, orimap)

def make_input_list(image_file_list): 
    """jpg 이미지 파일에서 ndarray로 변환

    Args:
        image_file_list (list): 이미지 파일 경로 모음 리스트 

    Returns:
        ndarray: ndarray 타입의 데이터로 구성된 ndarray 리스트 (5011, 224, 224, 3) -> (N, Height, Width, channel)
    """
    images_list = []
    
    for i in tqdm(range(0, len(image_file_list)), desc="get image"):
    
        image = cv2.imread(image_file_list[i])
        image = cv2.resize(image, (224, 224))/255
        
        images_list.append(image)
    
    return np.asarray(images_list)

def make_groundTruthBox_list(xml_file_path): 
    """xml파일에서 ground truth box 추출하는 함수  

    Args:
        xml_file_path (list): xml 파일 경로 리스트 

    Returns:
        ndarray: ndarray 타입의 데이터로 구성된 ndarray 리스트 (5011, 4) -> (N, box) box는 양끝 모서리 좌표로 이루어져 있다
    """

    f = open(xml_file_path)
    xml_file = xmltodict.parse(f.read()) 

    # 원본 이미지의 높이, 너비 
    Image_Height = float(xml_file['annotation']['size']['height'])
    Image_Width  = float(xml_file['annotation']['size']['width'])

    Ground_Truth_Box_list = [] 

    # multi-objects in image
    try:
        for obj in xml_file['annotation']['object']:
            
            # 박스 좌표(왼쪽 위, 오른쪽 아래) 얻기
            x_min = float(obj['bndbox']['xmin']) 
            y_min = float(obj['bndbox']['ymin'])
            x_max = float(obj['bndbox']['xmax']) 
            y_max = float(obj['bndbox']['ymax'])

            # 224x224에 맞게 변형
            x_min = float((224/Image_Width)*x_min)
            y_min = float((224/Image_Height)*y_min)
            x_max = float((224/Image_Width)*x_max)
            y_max = float((224/Image_Height)*y_max)

            Ground_Truth_Box = [x_min, y_min, x_max, y_max]
            Ground_Truth_Box_list.append(Ground_Truth_Box)

    # single-object in image
    except TypeError as e : 
        # 박스 좌표(왼쪽 위, 오른쪽 아래) 얻기
        x_min = float(xml_file['annotation']['object']['bndbox']['xmin']) 
        y_min = float(xml_file['annotation']['object']['bndbox']['ymin']) 
        x_max = float(xml_file['annotation']['object']['bndbox']['xmax']) 
        y_max = float(xml_file['annotation']['object']['bndbox']['ymax']) 

        # 224*224에 맞게 변형
        x_min = float((224/Image_Width)*x_min)
        y_min = float((224/Image_Height)*y_min)
        x_max = float((224/Image_Width)*x_max)
        y_max = float((224/Image_Height)*y_max)

        Ground_Truth_Box = [x_min, y_min, x_max, y_max]  
        Ground_Truth_Box_list.append(Ground_Truth_Box)

    
    Ground_Truth_Box_list = np.asarray(Ground_Truth_Box_list)
    Ground_Truth_Box_list = np.reshape(Ground_Truth_Box_list, (-1, 4))

    return Ground_Truth_Box_list


def make_class_list(xml_file_list):
    """이미지 데이터 상에 있는 물체 클래스를 리스트로 불러오는 함수 

    Args:
        xml_file_list (list): xml 파일 경로 리스트 

    Returns:
        list: 클래스로 이루어진 리스트 
    """
    Classes_inDataSet = []

    for xml_file_path in xml_file_list: 

        f = open(xml_file_path)
        xml_file = xmltodict.parse(f.read())
        # 사진에 객체가 여러개 있을 경우
        try: 
            for obj in xml_file['annotation']['object']:
                Classes_inDataSet.append(obj['name'].lower()) # 들어있는 객체 종류를 알아낸다
        # 사진에 객체가 하나만 있을 경우
        except TypeError as e: 
            Classes_inDataSet.append(xml_file['annotation']['object']['name'].lower()) 
        f.close()

    Classes_inDataSet = list(set(Classes_inDataSet))
    Classes_inDataSet.sort() 

    return Classes_inDataSet

def make_DataSet_forFastRCNN(xml_file_list, Classes_inDataSet):
    # Label List를 받아 데이터셋에 어떤 클래스가 있는지 알아내고 클래스 종류를 받아 이미지별 어떤 클래스가 있는지 Ground Truth Box별로 one-hot encoding을 해서 반환한다
    # 이미지별 GroundTruthBox도 반환한다
    num_Classes = len(Classes_inDataSet) # 데이터셋에 클래스 개수

    # 클래스 리스트를 알았으니 데이터셋을 만들어보자
    # 훈련 이미지 5011개 분의 데이터를 얻어야한다
    Cls_labels_for_FastRCNN = []
    Reg_labels_for_FastRCNN = []

    for i in tqdm(range(0, len(xml_file_list)), desc = "get_dataset_forFASTRCNN"):
        GroundTruthBoxes_inImage = make_groundTruthBox_list(xml_file_list[i]) # 이미지별 Ground Truth Box 리스트. (n, 4)크기의 리스트 받음

        classes = []
        f = open(xml_file_list[i])
        xml_file = xmltodict.parse(f.read())
        # 사진에 객체가 여러개 있을 경우
        try: 
            for obj in xml_file['annotation']['object']:
                classes.append(obj['name'].lower()) # 들어있는 객체 종류를 알아낸다
        # 사진에 객체가 하나만 있을 경우
        except TypeError as e:
            classes.append(xml_file['annotation']['object']['name'].lower()) 
        
        # 한 이미지에서 얻은 물체들의 각 클래스가 클래스 리스트 내에서 어떤 인덱스를 갖는지 확인하고 one-hot encoding을 수행한다 
        cls_index_list = []
        for class_val in classes :
            cls_index = Classes_inDataSet.index(class_val) # 클래스 리스트 내에서 어떤 인덱스인지 확인
            cls_index_list.append(cls_index)# 한 이미지 내에 있는 Ground Truth Box별로 갖고 있는 클래스 인덱스를 저장
        cls_onehot_inImage = np.eye(len(Classes_inDataSet)+1)[cls_index_list] # (n,21) 크기의 리스트로 각 Ground Truth Box당 클래스의 one-hot encoding된 라벨을 저장. 여기서 n은 한 이미지 내에 있는 객체 숫자

        # 저장
        Reg_labels_for_FastRCNN.append(GroundTruthBoxes_inImage)
        Cls_labels_for_FastRCNN.append(cls_onehot_inImage)

    return Reg_labels_for_FastRCNN, Cls_labels_for_FastRCNN # 이미지별 Ground Truth Box와 Classes 리스트

def generator_dataset_forFood(img_dir, annot_dir, model_path, image_folder_name):
    """학습을 위한 데이터 셋 구성

    Args:
        img_dir (string): 이미지 파일이 있는 폴더 경로 
        annot_dir (string): json 파일이 있는 폴더 경로 
        model_path (string): edge_detect.yml 파일이 있는 경로 
        image_folder_name (list): 클래스 별 폴더 이름 리스트 

    Yields:
        generator: 이미지 하나, 이미지에 해당하는 one-hot 인코딩된 클래스 리스트, Groundt truth box리스트, RoI 리스트 
    """
    for object in image_folder_name:
        for file in sorted(os.listdir(img_dir+object)):
            image_path = img_dir+object+'/'+file
            annot_path = annot_dir+object+'/'+file[:-3]+'json'
            # 이미지 파일, 어노테이션 파일 쌍이 존재하지 않는 경우 예외 처리 추가 필요 
            if file != '.DS_Store' and os.path.isfile(image_path) and os.path.isfile(annot_path):
                # Numpy 데이터로 변환 
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
                # image = cv2.resize(image, (448, 448))/255
                # image = image[np.newaxis]
                HEIGHT, WIDTH = image.shape[0], image.shape[1]
                # 2000개 RoI 뽑기  
                edgeboxes = EdgeBoxes(model_path)(image)
                rois, _ = edgeboxes
                 
                boxes = pd.read_json(annot_path)
                
                gtbox = []
                cls_index_list = []
                for box in boxes.iterrows():
                    x,y = box[1]['Point(x,y)'].split(',')
                    w,h = box[1]['W'],box[1]['H']
                    
                    x = float(x)*WIDTH
                    y = float(y)*HEIGHT
                    w = float(w)*WIDTH
                    h = float(h)*HEIGHT
                    if w == '' or h == '':
                        print("Except file", object, file, x)
                        continue
                    x, y = int(x - w/2), int(y - h/2)
                    w, h = int(w), int(h)
                    # Ground Truth box 저장 
                    gtbox.append([x,y,w,h])
                    # Class one-hot encoding
                    cls_index = image_folder_name.index(object) + 1 
                    cls_index_list.append(cls_index)
                cls_onehot = np.eye(len(image_folder_name)+1)[cls_index_list]        
                yield image.astype(np.float32), np.array(cls_onehot, dtype=np.uint32), \
                    np.array(gtbox, dtype=np.uint32), np.array(rois, dtype=np.uint32)

def get_iou(box1, box2):
        """(box1과 box2의 중첩되는 부분 / box1과 box2의 전체 영역)을 계산하는 함수 

        Args:
            box1: [x1, y1, x2, y2] / [왼쪽 아래 x1, 왼쪽 아래 y1, 오른쪽 위 x2, 오른쪽 위 y2]
            box2: [x1, y1, x2, y2] / [왼쪽 아래 x1, 왼쪽 아래 y1, 오른쪽 위 x2, 오른쪽 위 y2]
        Returns:
            Float : 0 ~ 1 사이 값 
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 겹칠 때
        if (x1 < x2 and y1 < y2):
            w = x2-x1 
            h = y2-y1 
            overlap_area =  w*h
        
        # 겹치지 않을 때
        else:  
            return 0

        area_b1 = (box1[2] - box1[0])*(box1[3] - box1[1])
        area_b2 = (box2[2] - box2[0])*(box2[3] - box2[1])
        union =  area_b1 + area_b2 - overlap_area

        return overlap_area/union

def get_minibatch(RoI_list, GTB, Target_object): 
        """
        로스를 계산하기 전에 입력받은 데이터에서 RoI 64개씩 추출
        Args:
            RoI_list (numpy.ndarray): 최대 2000개 roi 
            GTB (numpy.ndarray): 이미지 하나에 해당하는 Ground truth (x_min, y_min, x_max, y_max) 
            Target_object (numpy.ndarray): 이미지 하나에 해당하는 target class (one hot encoding)

        Returns:
            RoI_minibatch, Target_minibatch, GTB_minibatch, positive_num: 64개 RoI, 64개 정답 라벨 데이터, 64개 정답 박스 데이터, positive sample 개수 
        """
        positive_RoI_list = []
        positive_target = []
        positive_GTB = []

        negative_RoI_list = []
        negative_target = []
        negative_GTB = []

        for i in range(0, len(GTB)):
            ground_truth_box = GTB[i]
            target = Target_object[i]
            
            for j in range(0, len(RoI_list)):
                # roi 좌표는 x,y,w,h로 구성되어 있기 때문에 양끝 좌표로 변환
                RoI_xmin, RoI_ymin, RoI_xmax, RoI_ymax = convert_to_corners(RoI_list[j], mode='corner')
                gtb_xmin, gtb_ymin, gtb_xmax, gtb_ymax = convert_to_corners(ground_truth_box, mode='corner')
                
                IoU = get_iou([RoI_xmin, RoI_ymin, RoI_xmax, RoI_ymax], \
                              [gtb_xmin, gtb_ymin, gtb_xmax, gtb_ymax])
        
                # 물체라고 생각되는 것들 
                if IoU >= 0.3 :
                    positive_RoI_list.append([RoI_xmin, RoI_ymin, RoI_xmax, RoI_ymax])
                    positive_target.append(target)
                    positive_GTB.append(ground_truth_box)
                # 배경이라고 생각되는 것들
                elif IoU >= 0.1 and IoU < 0.4:
                    negative_RoI_list.append([RoI_xmin, RoI_ymin, RoI_xmax, RoI_ymax])
                    negative_target.append(np.zeros(len(target)))
                    negative_GTB.append([0,0,0,0])
                    
                    
        # positive sample이 16이하일 때는 어떻게 할까             
        positive_num = 16 # min([16, len(positive_RoI_list)])
        
        RoI_minibatch = random.sample(list(positive_RoI_list), positive_num)
        Target_minibatch = random.sample(list(positive_target), positive_num)
        GTB_minibatch = random.sample(list(positive_GTB), positive_num)
        
        RoI_minibatch.extend(random.sample(list(negative_RoI_list), 64-positive_num))
        Target_minibatch.extend(random.sample(list(negative_target), 64-positive_num))
        GTB_minibatch.extend(random.sample(list(negative_GTB), 64-positive_num))

        return np.array(RoI_minibatch), np.array(Target_minibatch), \
            np.array(GTB_minibatch), positive_num


def change_folder_name(path, category):
    """폴더명을 일괄적으로 변경하는 함수

    Args:
        path (string): 이름을 바꿀 폴더가 모여있는 폴더의 상위 폴더 경로
        category (list): 이름을 바꿀 폴더 리스트 
    Returns:
        None
    """
    for folder in os.listdir(path):
        if os.path.isdir(path+folder):
            if 'json' in folder:
                folder_name = folder.repace('json','').replace('.','').strip()
                os.rename(path+folder, path+category[folder_name])
            else:
                os.rename(path+folder, path+category[folder])
        else:
            print(folder)

def folderName_to_textfile(path, filename):
    folder = os.listdir(path)
    [folder.remove(x) if '.' in x else x for x in folder]
    with open(filename, 'w+') as f:
        f.write('\n'.join(folder))
    

            
def convert_to_corners(boxes, mode='center'):
        """박스 형식을 모서리쪽 좌표로 바꿈

        Args:
            boxes : [x, y, width, height] / [모서리 좌표 x, 모서리 좌표 y, 넓이, 높이]
            mode : center => x, y 좌표쌍이 박스의 중심인 경우 
                   corner => x, y 좌표쌍이 박스의 모서리인 경우 
        Returns:
            [] : [x1, y2, x2, y2] / 대각선 모서리 좌표 쌍 
        """
        if mode == 'center': 
            x1 = boxes[0] - (boxes[2] / 2)
            y1 = boxes[1] - (boxes[3] / 2)
            x2 = boxes[0] + (boxes[2] / 2)
            y2 = boxes[1] + (boxes[3] / 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            return x1, y1, x2, y2
        elif mode == 'corner':
            x1 = boxes[0] 
            y1 = boxes[1] 
            x2 = boxes[0] + boxes[2]
            y2 = boxes[1] + boxes[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            return x1, y1, x2, y2
        
def remove_file(path, image_folder_name, file_name):
    for object in image_folder_name:
        for file in os.listdir(path+object):
            if file == file_name:
                os.remove(path+object+'/'+file)
                