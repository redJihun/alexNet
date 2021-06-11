# For files
import os
import time

# Deep-learning framework
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# Manipulate
import numpy as np
import random

RANDOM_SEED = 602
# random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Hyper-parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005         # == weight_decay
LR_INIT = 0.01            # == weight_init
IMAGE_DIM = 227
NUM_CLASSES = 5
IMAGENET_MEAN = np.array([104., 117., 124.], dtype=np.float)

# Data directory
INPUT_ROOT_DIR = './input'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
OUTPUT_ROOT_DIR = './output'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


########################################################################################################################
# @todo Load dataset
# After init variables, append imagefile's path and label(in number, origin is name of sub-directory).
# Set the path of root dir, and use os.walk(root_dir) for append all images in sub-dir.
def load_dataset(path):
    imagepaths, labels = list(), list()
    label = 0
    classes = sorted(os.walk(path).__next__()[1])
    for c in classes:
        c_dir = os.path.join(path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image
        for sample in walk[2]:
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
        # next directory
        label += 1

    # Read images from disk
    # Resize & Crop
    print('start resizing image')
    images = list()
    for img in imagepaths:
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=(227, 227))
        # img = tf.image.crop_and_resize(tf.reshape(img, shape=(-1,256,256,3)), crop_size=(227, 227), boxes=[5, 4], box_indices=[5, ])
        images.append(img)
    print('end resizing')
    # images = tf.data.Dataset.from_tensors(tf.convert_to_tensor(images, dtype=tf.float32))
    # img = tf.image.crop_and_resize(img, crop_size=(227, 227), boxes=[900., 4.], box_indices=[900, ])
    # image = tf.image.crop_and_resize(image, crop_size=(227,227), boxes=[900, 4])

    # Shuffle with seed can keep the data-label pair. Without shuffle, data have same label in range.
    foo = list(zip(images, labels))
    random.Random(RANDOM_SEED).shuffle(foo)
    # random.Random(RANDOM_SEED).shuffle(images)
    # random.Random(RANDOM_SEED).shuffle(labels)
    images, labels = zip(*foo)

    # Split train/valid/test, total data size = 5,000
    train_X, train_Y = images[:int(len(images)*0.8)], labels[:int(len(labels)*0.8)]
    valid_X, valid_Y = images[int(len(images)*0.8):int(len(images)*0.9)], labels[int(len(labels)*0.8):int(len(labels)*0.9)]
    test_X, test_Y = images[int(len(images)*0.9):], labels[int(len(labels)*0.9):]

    # Convert to Tensor
    train_X, train_Y = tf.convert_to_tensor(train_X, dtype=tf.float32), tf.convert_to_tensor(train_Y, dtype=tf.int32)
    valid_X, valid_Y = tf.convert_to_tensor(valid_X, dtype=tf.float32), tf.convert_to_tensor(valid_Y, dtype=tf.int32)
    test_X, test_Y = tf.convert_to_tensor(test_X, dtype=tf.float32), tf.convert_to_tensor(test_Y, dtype=tf.int32)

    # Build Tf dataset
    train_X, train_Y = tf.data.Dataset.from_tensor_slices(tensors=train_X).batch(batch_size=128), tf.data.Dataset.from_tensor_slices(tensors=train_Y).batch(batch_size=128)
    valid_X, valid_Y = tf.data.Dataset.from_tensor_slices(tensors=valid_X), tf.data.Dataset.from_tensor_slices(tensors=valid_Y)
    test_X, test_Y = tf.data.Dataset.from_tensor_slices(tensors=test_X), tf.data.Dataset.from_tensor_slices(tensors=test_Y)

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y
########################################################################################################################


########################################################################################################################
# 원 라벨 y와 예측된 라벨을 비교하여 계산된 loss를 리턴하는 함수
# 모델 구조가 포함되었음
# 매개변수로 현재 수행 중인 epoch(=step), 입력된 이미지 데이터(=x), 입력된 데이터의 라벨(=y), 가중치 dictionary(=param) 가 주어짐
def loss(step, x, y, param):
    inputs = tf.constant(x, name='inputs')

    # layer 1
    l1_convolve = tf.nn.conv2d(input=inputs, filters=param['w1'], strides=4, padding='VALID', name='l1_convolve')
    l1_bias = tf.reshape(tf.nn.bias_add(l1_convolve, param['b1']), tf.shape(l1_convolve), name='l1_bias')
    l1_relu = tf.nn.relu(l1_bias, name='l1_relu')
    l1_norm = tf.nn.lrn(input=l1_relu, depth_radius=5, alpha=10e-4, beta=0.75, bias=2.0, name='l1_norm')
    l1_pool = tf.nn.max_pool(input=l1_norm, ksize=3, strides=2, padding='VALID', name='l1_pool')

    # layer 2
    l2_convolve = tf.nn.conv2d(input=l1_pool, filters=param['w2'], strides=1, padding='SAME', name='l2_convolve')
    l2_bias = tf.reshape(tf.nn.bias_add(l2_convolve, param['b2']), tf.shape(l2_convolve), name='l2_bias')
    l2_relu = tf.nn.relu(l2_bias, name='l2_relu')
    l2_norm = tf.nn.lrn(input=l2_relu, depth_radius=5, alpha=10e-4, beta=0.75, bias=2.0, name='l2_norm')
    l2_pool = tf.nn.max_pool(input=l2_norm, ksize=3, strides=2, padding='VALID', name='l2_pool')

    # layer 3
    l3_convolve = tf.nn.conv2d(input=l2_pool, filters=param['w3'], strides=1, padding='SAME', name='l3_convolve')
    l3_bias = tf.reshape(tf.nn.bias_add(l3_convolve, param['b3']), tf.shape(l3_convolve), name='l3_bias')
    l3_relu = tf.nn.relu(l3_bias, name='l3_relu')

    # layer 4
    l4_convolve = tf.nn.conv2d(input=l3_relu, filters=param['w4'], strides=1, padding='SAME', name='l4_convolve')
    l4_bias = tf.reshape(tf.nn.bias_add(l4_convolve, param['b4']), tf.shape(l4_convolve), name='l4_bias')
    l4_relu = tf.nn.relu(l4_bias, name='l4_relu')

    # layer 5
    l5_convolve = tf.nn.conv2d(input=l4_relu, filters=param['w5'], strides=1, padding='SAME', name='l5_convolve')
    l5_bias = tf.reshape(tf.nn.bias_add(l5_convolve, param['b5']), tf.shape(l5_convolve), name='l5_bias')
    l5_relu = tf.nn.relu(l5_bias, name='l5_relu')
    l5_pool = tf.nn.max_pool(input=l5_relu, ksize=3, strides=2, padding='VALID', name='l5_pool')

    # layer 6
    l6_flattened = tf.reshape(l5_pool, [-1, tf.shape(param['w6'])[0]], name='l6_flattened')
    l6_fc = tf.nn.bias_add(tf.matmul(l6_flattened, param['w6']), param['b6'], name='l6_fc')
    l6_relu = tf.nn.relu(l6_fc, name='l6_relu')
    l6_dropout = tf.nn.dropout(l6_relu, rate=0.5, name='l6_dropout')

    # layer 7
    l7_fc = tf.nn.bias_add(tf.matmul(l6_dropout, param['w7']), param['b7'], name='l7_fc')
    l7_relu = tf.nn.relu(l7_fc, name='l7_relu')
    l7_dropout = tf.nn.dropout(l7_relu, rate=0.5, name='l7_dropout')

    # layer 8
    logits = tf.nn.bias_add(tf.matmul(l7_dropout, param['w8']), param['b8'], name='l8_fc')
    # 출력된 logits는 라벨을 예측한 형태가 아닌 각 클래스에 대한 값이 존재하는 형태, shape[데이터 개수, 클래스 개수]
    # argmax를 적용함으로써, 단일 라벨을 출력하는 형태로 변경, y(ground_truth) 와 비교해 loss 계산
    predict = tf.argmax(logits, 1).numpy()

    # 멀티 라벨로 모델 평가 시에 수렴되지 않는 문제 발생(추가적인 연구 필요, 아마 데이터의 클래스 개수가 너무 적어 그런것으로 예상)
    # 단일 라벨로 평가(sparse_softmax_cross_entropy_with_logits)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict)
    loss = tf.reduce_mean(loss)
    target = y

    accuracy = np.sum(predict == target) / len(target)

    print('epoch {}:\tloss = {}\taccuracy = {}'.format(step, loss.numpy(), accuracy))

    return loss
########################################################################################################################


########################################################################################################################
# 논문 상에서 loss가 진동 시 learning_rate를 10으로 나누어주는 역할
# 적용 방법의 추가적인 연구가 필요.
# lr_schedule = tf.optimizers.schedules.De(
#     initial_learning_rate= 0.001,
#     decay_steps=
# )

# 만들어준 모델에서 back-prop 과 가중치 업데이트를 수행하기 위해 optimizer 메소드를 사용
# 기존 텐서플로우에는 weight-decay 가 설정 가능한 optimizer 부재, Tensorflow_addons 의 SGDW 메소드 사용
# learning_rate를 0.01(follow 논문)으로 설정 시, loss가 발산하는 문제 발생, 따라서 0.001로 설정
# optimizer = tfa.optimizers.SGDW(momentum=MOMENTUM, learning_rate=0.001, weight_decay=0.5, name='optimizer')
optimizer = tfa.optimizers.SGDW(momentum=MOMENTUM, learning_rate=0.001, weight_decay=LR_DECAY, name='optimizer')
# optimizer = tf.optimizers.SGD(learning_rate=LR_INIT, name='optimizer')

# 파라미터(=가중치) 들을 직접 관리해야 하므로 논문 조건에 따라 미리 초기화를 수행해줌
parameters = {
    'w1': tf.Variable(tf.random.normal(shape=[11, 11, 3, 96], mean=0.0, stddev=0.01, dtype=tf.float32), name='w1', trainable=True),
    'b1': tf.Variable(tf.zeros(shape=[96]), trainable=True, name='b1'),

    'w2': tf.Variable(tf.random.normal(shape=[5, 5, 96, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w2', trainable=True),
    'b2': tf.Variable(tf.ones(shape=[256]), trainable=True, name='b2'),

    'w3': tf.Variable(tf.random.normal(shape=[3, 3, 256, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w3', trainable=True),
    'b3': tf.Variable(tf.zeros(shape=[384]), trainable=True, name='b3'),

    'w4': tf.Variable(tf.random.normal(shape=[3, 3, 384, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w4', trainable=True),
    'b4': tf.Variable(tf.ones(shape=[384]), trainable=True, name='b4'),

    'w5': tf.Variable(tf.random.normal(shape=[3, 3, 384, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w5', trainable=True),
    'b5': tf.Variable(tf.ones(shape=[256]), trainable=True, name='b5'),

    'w6': tf.Variable(tf.random.normal(shape=[6*6*256, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w6', trainable=True),
    'b6': tf.Variable(tf.ones(shape=[4096]), trainable=True, name='b6'),

    'w7': tf.Variable(tf.random.normal(shape=[4096, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w7', trainable=True),
    'b7': tf.Variable(tf.ones(shape=[4096]), trainable=True, name='b7'),

    'w8': tf.Variable(tf.random.normal(shape=[4096, NUM_CLASSES], mean=0.0, stddev=0.01, dtype=tf.float32), name='w8', trainable=True),
    'b8': tf.Variable(tf.zeros(shape=[NUM_CLASSES]), trainable=True, name='b8'),
}

# 사전에 정의한 load_dataset 함수의 매개변수로 이미지를 저장한 파일경로의 루트 디렉토리 지정
train_X, train_Y, valid_X, valid_Y, test_X, test_Y = load_dataset(TRAIN_IMG_DIR)

# 정해진 횟수(90번)만큼 training 진행 -> 전체 트레이닝셋을 90번 반복한다는 의미
for epoch in range(NUM_EPOCHS):
    # 몇 번째 batch 수행 중인지 확인 위한 변수
    foo = 1
    # batch_size(128)로 나뉘어진 데이터에서 트레이닝 수행, e.g., 2000개의 데이터 / 128 = 15.625 -> 16개의 batch
    # 즉 16번 가중치 업데이트가 이루어짐
    for batch_X, batch_Y in zip(list(train_X.as_numpy_iterator()), list(train_Y.as_numpy_iterator())):
        print('batch {}'.format(foo))
        foo += 1
        # loss 함수의 정의에 따라 feed-forward 과정 수행, minimize 메소드로 back-prop 수행 & 가중치 업데이트
        # 현재 가중치를 직접 관리하는 중, 따라서 직접 초기화 수행 후 매개변수로 가중치 딕셔너리를 넣어줌
        optimizer.minimize(lambda: loss(epoch, batch_X, batch_Y, parameters), var_list=parameters)

# Save the updated parameters(weights, biases)
np.savez(os.path.join(CHECKPOINT_DIR, 'trained_parameters'+time.strftime('%y%m%d%H%M%S', time.localtime())), parameters)
