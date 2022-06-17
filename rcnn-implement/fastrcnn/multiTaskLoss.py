import tensorflow as tf 
import numpy as np

def multi_task_loss(predict_object_output, predict_box_output, target_object, ground_truth_box, positive_num):  
    """Regressor와 Classifier의 loss를 동시에 구하는 함수 

    Args:
        predict_object_output (Tensor): FastRCNN 모델의 Classifier 순전파 결과 
                                        (feature vector/[None, RoI개수, Object 개수, Class 개수 + 1])
                                         ex) (2, 64, 2, 11) Batch 2, Positive + Negative RoI, 물체 개수, 클래스(10) + 배경(1)
        predict_box_output (Tensor): FastRCNN 모델의 Regressor 순전파 결과 
                                        (feature vector/[None, RoI개수, Object 개수, 4]) 
                                         ex) (2, 16, 2, 4) Batch 2, Positive RoI, 물체 개수, x,y,w,h
        target_object (ndarray): 실제 물체의 one-hot encoding된 라벨 
        ground_truth_box (ndarray): 실제 물체에 해당하는 박스 좌표 
        positive_num (int): positive sample 개수 (Regressor 학습시 positive sample만 진행하기 위해)

    Returns:
        loss (Tensor): 누적된 loss
    """
    # class_index = [tf.argmax(x).numpy() for x in target_object]
    # pred_box = [predict_box_output[0, x, 4*x:4*x + 4] for x in class_index]
    # print(pred_box,len(pred_box),pred_box[0].shape)
    # reg_loss = tf.compat.v1.losses.huber_loss(labels=ground_truth_box[:16,...], predictions=predict_box_output[0,:16,...])
    
    cls_loss = tf.compat.v1.losses.log_loss(labels=target_object, predictions=predict_object_output[0,...])
    reg_loss = tf.compat.v1.losses.huber_loss(labels=ground_truth_box, predictions=predict_box_output[0,...])
    total_loss = tf.add(cls_loss, reg_loss)
    return total_loss