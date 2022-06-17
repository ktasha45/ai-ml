import os
from pickletools import optimize
from numpy import gradient 

import tensorflow as tf 
from FastRCNN import FastRCNN
from dataUtils import generator_dataset_forFood
from dataUtils import get_minibatch
from multiTaskLoss import multi_task_loss

CURRENT_DIR = os.getcwd()

IMG_DIR = CURRENT_DIR+'/data/train/'
ANNOT_DIR = CURRENT_DIR+'/data/train_json/'
MODEL_PATH = CURRENT_DIR+'/edge_detect.yml'

# 음식 데이터 
image_folder_name = [x for x in os.listdir(IMG_DIR)]
image_folder_name = sorted(image_folder_name)
image_folder_name.remove('.DS_Store')

IMG_SIZE = (448, 448)
BATCH_SIZE = 1
idg = tf.keras.preprocessing.image.ImageDataGenerator()
data = idg.flow_from_directory(directory=IMG_DIR,
                       target_size=IMG_SIZE,
                       class_mode='categorical',
                       batch_size=BATCH_SIZE,
                       shuffle=True,)

image_sizes = data.n

# Generator 형식의 데이터로 호출 할 때에만 메모리에 적재된다. 
# image, one-hot-class, Ground Truth box, RoIs

train_datasets = tf.data.Dataset.from_generator(lambda : generator_dataset_forFood(IMG_DIR, ANNOT_DIR, MODEL_PATH, image_folder_name),
                               output_types=((tf.float32), (tf.uint32), (tf.uint32), (tf.uint32)),
                               output_shapes=(((None, None, 3)), ((None, 5)), ((None,4)), ((None,4))))


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
train_loss = tf.keras.metrics.Mean(name="train_loss")
precision = tf.keras.metrics.Precision(name="precision")
recall = tf.keras.metrics.Recall(name="recall")


def train_step(image, target_object, gtb, rois):
    # mini_target_object (64, 5) / mini_gtb (64,4)
    mini_rois, mini_target_object, mini_gtb, positive_num = get_minibatch(rois[0,...],gtb[0,...], target_object[0,...])
    with tf.GradientTape() as tape:    
        predict_object, predict_box = fastrcnn(image, mini_rois) # (1, 64, 5) (1, 64, 4)
        losses = multi_task_loss(predict_object, predict_box, mini_target_object, mini_gtb, positive_num)
    gradients = tape.gradient(losses, fastrcnn.trainable_weights)
    optimizer.apply_gradients(zip(gradients, fastrcnn.trainable_weights))
    train_loss(losses)

def train_model(dataset, batch, epochs, ckpt, manager, patience):
    """학습하기 전에 데이터를 섞고 배치 단위로 불러와서 학습을 한다 

    Args:
        dataset (generator): tf.data.Dataset으로 구성된 데이터 셋
        batch (int): 1 iteration동안 사용할 데이터 수(양의 정수)
        epochs (int): 전체 데이터를 몇번 반복할지 정하는 횟수(양의 정수) 
        ckpt (Checkpoint): 지정한 학습 step 마다 학습 파라미터를 저장한다 
        manager (CheckpointManager): 다수의 checkpoint를 관리할 때 사용한다 
        patience (int): Loss가 증가할 때 얼만큼 기다렸다 학습을 종료할지 
    """
    patience = patience
    wait = 0
    before = 9999
    
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    cnt=0
    template = 'epoch: {}, train_loss: {:.4f}'
    
    for image, target_object, gtb, rois \
        in dataset.shuffle(10).batch(batch).repeat(epochs):
        train_step(image, target_object, gtb, rois)
        ckpt.step.assign_add(1) 
        cnt+=1
        print(template.format(cnt, train_loss.result()))
        
        if int(ckpt.step) % 10 == 0: 
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        
        # early stoppling
        if train_loss.result() < before:
            before = train_loss.result()
            wait = 0
        else: 
            before = train_loss.result()
            wait += 1
        if wait >= patience:
            print("Early stopping")
            break
    

vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
SharedConvNet = tf.keras.models.Sequential()
for layer in vgg.layers[:-1]:
    SharedConvNet.add(layer)


fastrcnn = FastRCNN(SharedConvNet)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=fastrcnn)
manager = tf.train.CheckpointManager(ckpt, './ckpts', max_to_keep=3)

train_model(train_datasets, 1, 1, ckpt, manager, 5)