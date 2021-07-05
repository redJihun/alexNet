# For files
import os
import time

# Deep-learning framework
import cv2
import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# Manipulate
import numpy as np
import random

from sklearn.preprocessing import minmax_scale

RANDOM_SEED = 602
# random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

# Hyper-parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005         # == weight_decay
LR_INIT = 0.01
NUM_CLASSES = 3
IMAGENET_MEAN = np.array([50., 50., 50.], dtype=np.float)

# Data directory
INPUT_ROOT_DIR = './input/task'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
OUTPUT_ROOT_DIR = './output/task'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'train')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


########################################################################################################################
# 이미지 로드 및 전처리 + 데이터 증강 함수들

# After init variables, append imagefile's path and label(in number, origin is name of sub-directory).
# Set the path of root dir, and use os.walk(root_dir) for append all images in sub-dir.
def load_imagepaths(path):
    imagepaths, labels = list(), list()
    label = 0
    classes = os.walk(path).__next__()[1]
    for c in classes:
        c_dir = os.path.join(path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image
        for sample in walk[2]:
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
        # next directory
        label += 1

    # 32개 데이터씩 split 하므로 미리 원 데이터를 셔플해준 후에 증강 진행.
    foo = list(zip(imagepaths, labels))
    random.shuffle(foo)
    imagepaths, labels = zip(*foo)

    return imagepaths, labels


# 256x256으로 이미지 다운샘플링
def resize_images(imgpaths):
    # Read images from disk & resize
    # print('start resizing image')
    images = list()
    for img in imgpaths:
        image = cv2.imread(img)
        image = cv2.resize(image, dsize=(227, 227), interpolation=cv2.INTER_AREA)
        images.append(image)
    # print('end resizing')
    return images


# RGB jittering
def fancy_pca(images, labels, alpha_std=0.1):
    # print('Start Jittering')
    pca_images, pca_labels = images.copy(), labels.copy()
    for img, lbl in zip(images, labels):
        orig_img = np.array(img, dtype=np.float).copy()

        # 이미지 픽셀값에서 이미지넷 평균 픽셀값을 빼줌(평균 픽셀값은 사전에 정의됨)
        img_rs = np.reshape(img, (-1, 3))
        img_centered = img_rs - IMAGENET_MEAN

        # 해당 이미지의 공분산 행렬 구함
        img_cov = np.cov(img_centered, rowvar=False)

        # 고유벡터, 고유값 구함
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]

        # 고유벡터 세개를 쌓아서 3x3 행렬로 만듦
        m1 = np.column_stack((eig_vecs))
        m2 = np.zeros((3, 1))

        # 랜덤값과 고유값을 곱함
        alpha = np.random.normal(0, alpha_std)
        m2[:, 0] = alpha * eig_vals

        # 3x3 고유벡터 행렬과 3x1 랜덤값*고유값 행렬을 곱해서 3x1 행렬을 얻음(RGB 채널에 가감해줄 값)
        add_vect = np.matrix(m1) * np.matrix(m2)

        # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
        for idx in range(3):
            orig_img[..., idx] += add_vect[idx]
            # minmax_scale(orig_img[..., idx], feature_range=(0., 1.), copy=False)

        # 0~255(rgb픽셀값) 범위로 값 재설정
        pca_img = orig_img
        pca_images.append(pca_img)
        pca_labels.append(lbl)
    # print('End jittering')
    return pca_images, pca_labels


def minmax(images, min, max):
    scaled_images = list()
    for img in images:
        # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
        scaled_img = np.array(img).copy()
        for idx in range(3):
            scaled_img[..., idx] = minmax_scale(img[..., idx], feature_range=(min, max))
        scaled_images.append(scaled_img)

    return scaled_images


# horizontal reflection
def flip_image(images, labels):
    # print('Start flipping')
    flipped_images, flipped_labels = images.copy(), labels.copy()
    for img, lbl in zip(images, labels):
        flipped_image = tf.image.flip_left_right(img)
        flipped_images.append(flipped_image)
        flipped_labels.append(lbl)
    # print('End flipping')
    return flipped_images, flipped_labels


# Image cropping
def crop_image(images, labels):
    # print('Start cropping')
    cropped_images, cropped_labels = list(), list()
    for img, label in zip(images, labels):
        # # left-top
        # cropped_img = tf.image.crop_to_bounding_box(img, 0, 0, 227, 227)
        # cropped_images.append(cropped_img)
        # cropped_labels.append(label)
        # # right-top
        # cropped_img = tf.image.crop_to_bounding_box(img, np.shape(img)[0]-227, 0, 227, 227)
        # cropped_images.append(cropped_img)
        # cropped_labels.append(label)
        # center
        cropped_img = tf.image.crop_to_bounding_box(img, int((np.shape(img)[0]-227)/2-1), int((np.shape(img)[0]-227)/2-1), 227, 227)
        cropped_images.append(cropped_img)
        cropped_labels.append(label)
        # # left-bottom
        # cropped_img = tf.image.crop_to_bounding_box(img, 0, np.shape(img)[0]-227, 227, 227)
        # cropped_images.append(cropped_img)
        # cropped_labels.append(label)
        # # right-bottom
        # cropped_img = tf.image.crop_to_bounding_box(img, np.shape(img)[0]-228, np.shape(img)[1]-228, 227, 227)
        # cropped_images.append(cropped_img)
        # cropped_labels.append(label)
    # print('End cropping')
    return cropped_images, cropped_labels


# 증강된 데이터를 입력받아 셔플 후 TF 데이터셋으로 리턴
def make_dataset(images, labels):
    # print('Start making dataset')
    # Shuffle with seed can keep the data-label pair. Without shuffle, data have same label in range.
    # foo = list(zip(images, labels))
    # random.Random(RANDOM_SEED).shuffle(foo)
    # images, labels = zip(*foo)

    # Convert to Tensor
    train_X, train_Y = tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)

    # Build Tf dataset
    train_X, train_Y = tf.data.Dataset.from_tensor_slices(tensors=train_X).batch(batch_size=BATCH_SIZE), tf.data.Dataset.from_tensor_slices(tensors=train_Y).batch(batch_size=BATCH_SIZE)
    # print('End making dataset')

    return train_X, train_Y
########################################################################################################################


########################################################################################################################
# 원 라벨 y와 예측된 라벨을 비교하여 계산된 loss, accuracy 리턴하는 함수
# 모델 구조가 포함되었음
# 매개변수로 현재 수행 중인 epoch(=step), 입력된 이미지 데이터(=x), 입력된 데이터의 라벨(=y), 가중치 dictionary(=param) 가 주어짐
def loss(batch_num, x, y, param, step, epoch):
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
    predict = tf.argmax(tf.nn.softmax(logits, 1), 1).numpy()

    threshold = 0.5
    softmax_scores = tf.nn.softmax(logits=logits)
    other_class_idx = tf.cast(tf.shape(softmax_scores)[0] + 1, tf.int64)
    other_class_idx = tf.tile(tf.expand_dims(other_class_idx, 0), [tf.shape(softmax_scores)[0]])

    # 멀티 라벨로 모델 평가 시에 수렴되지 않는 문제 발생(추가적인 연구 필요, 아마 데이터의 클래스 개수가 너무 적어 그런것으로 예상)
    # 단일 라벨로 평가(sparse_softmax_cross_entropy_with_logits)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, depth=NUM_CLASSES), logits=logits)
    # loss = tf.keras.losses.hinge(tf.one_hot(y, depth=NUM_CLASSES), tf.nn.softmax(logits, 1))
    # loss = tf.keras.losses.categorical_hinge(tf.one_hot(y, depth=NUM_CLASSES), tf.nn.softmax(logits, 1))
    loss = tf.reduce_mean(loss)

    accuracy = np.sum(predict == y) / len(y)

    print('epoch {}\tstep {}\tbatch {}\t:\tloss={}\taccuracy={}'.format(epoch, step, batch_num, loss.numpy(), accuracy))

    return loss
########################################################################################################################


########################################################################################################################
# 파라미터(=가중치) 들을 직접 관리해야 하므로 논문 조건에 따라 초기화를 수행하는 함수
def init_params():
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
    return parameters


########################################################################################################################


########################################################################################################################



def train(step, loop, imgs_path=TRAIN_IMG_DIR, epochs=NUM_EPOCHS):
    current_ckpt = os.path.join(OUTPUT_ROOT_DIR, str(loop))
    os.makedirs(current_ckpt, exist_ok=True)
    # 논문 상에서 loss가 진동 시 learning_rate를 10으로 나누어주는 역할
    # 적용 방법의 추가적인 연구가 필요.
    # lr_scheduler = tf.optimizers.schedules.PolynomialDecay(
    #     initial_learning_rate=LR_INIT,
    #     decay_steps=1000,
    #     end_learning_rate=LR_INIT/(10**3)
    # )
    # 만들어준 모델에서 back-prop 과 가중치 업데이트를 수행하기 위해 optimizer 메소드를 사용
    # 기존 텐서플로우에는 weight-decay 가 설정 가능한 optimizer 부재, Tensorflow_addons 의 SGDW 메소드 사용
    lr_temp = LR_INIT / 10
    # optimizer = tfa.optimizers.SGDW(momentum=MOMENTUM, learning_rate=lr_temp, weight_decay=LR_DECAY, name='optimizer')      # loss: 1.08... acc: 0.33
    # optimizer = tf.optimizers.SGD(momentum=MOMENTUM, learning_rate=0.001, name='optimizer')
    # optimizer = tf.optimizers.RMSprop(momentum=MOMENTUM, learning_rate=0.001, name='RMSprop')     #
    optimizer = tf.optimizers.Adam(learning_rate=lr_temp)

    # 파라미터(=가중치) 들을 직접 관리해야 하므로 논문 조건에 따라 초기화
    parameters = init_params()

    # 사전에 정의한 load_imagepaths 함수의 매개변수로 이미지를 저장한 파일경로의 루트 디렉토리 지정
    filepaths, labels = load_imagepaths(imgs_path)

    # 정해진 횟수(90번)만큼 training 진행 -> 전체 트레이닝셋을 90번 반복한다는 의미
    for epoch in range(epochs):
        start_time = time.time()
        print('epoch {}'.format(epoch+1))
        # 몇 번째 batch 수행 중인지 확인 위한 변수
        foo = 1
        min_loss = 99999999
        epoch_best_param = {}
        # batch_size(128)로 나뉘어진 데이터에서 트레이닝 수행, e.g., 2000개의 데이터 / 128 = 15.625 -> 16개의 batch
        # 즉, 1epoch에 16번 가중치 업데이트가 이루어짐
        for i in range(int(np.ceil(len(filepaths)/BATCH_SIZE))):
            # 마지막 split은 전체 데이터 개수가 32로 안 나누어 떨어지는 경우 남은 개수만큼만 로드
            if i == int(np.ceil(len(filepaths) / BATCH_SIZE)) - 1:
                fpaths, lbls = filepaths[i * BATCH_SIZE:], list(labels[i * BATCH_SIZE:])
            # 그 외의 split은 32의 배수로 나누어서 로드
            else:
                fpaths, lbls = filepaths[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], list(labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])

            imgs = resize_images(fpaths)
            # imgs, lbls = flip_image(imgs, lbls)
            # imgs, lbls = crop_image(imgs, lbls)
            # imgs, lbls = fancy_pca(imgs, lbls)
            imgs = minmax(imgs, -1.0, 1.0)
            train_X, train_Y = make_dataset(imgs, lbls)

            # batch_size(128)로 나뉘어진 데이터에서 트레이닝 수행, e.g., 2000개의 데이터 / 128 = 15.625 -> 16개의 batch
            # -> 1epoch에 16번 가중치 업데이트가 이루어짐
            for batch_X, batch_Y in zip(list(train_X.as_numpy_iterator()), list(train_Y.as_numpy_iterator())):
                # loss 함수의 정의에 따라 feed-forward 과정 수행, minimize 메소드로 back-prop 수행 & 가중치 업데이트
                # 현재 가중치를 직접 관리하는 중, 따라서 직접 초기화 수행 후 매개변수로 가중치 딕셔너리를 넣어줌
                # current_loss = loss(foo, batch_X, batch_Y, parameters, step, epoch+1)
                optimizer.minimize(lambda :loss(foo, batch_X, batch_Y, parameters, step, epoch+1), var_list=parameters)
                if step % 1000 == 0:
                    np.savez(os.path.join(current_ckpt, time.strftime('%y%m%d_%H%M', time.localtime()) + '_{}steps'.format(step)), parameters)
                foo += 1
                step += 1
                # if min_loss > current_loss:
                #     min_loss = current_loss
                    # epoch_best_param = parameters.copy()
        if (epoch+1) % 2 == 0 and lr_temp >= 1e-5:
            lr_temp /= 10;
            # optimizer = tfa.optimizers.SGDW(momentum=MOMENTUM, learning_rate=lr_temp, weight_decay=LR_DECAY, name='optimizer')
            # optimizer = tf.optimizers.RMSprop(momentum=MOMENTUM, learning_rate=lr_temp, name='RMSprop')
            optimizer = tf.optimizers.Adam(learning_rate=lr_temp)
    # Save the updated parameters(weights, biases)
    #     if (epoch+1) % 10 == 0:
    #     if (epoch + 1) % 2 == 0:
        np.savez(os.path.join(current_ckpt, time.strftime('%y%m%d_%H%M', time.localtime()) + '_{}epoch'.format(epoch+1)), parameters)
        # parameters = epoch_best_param.copy()
        end_time = time.time()
        print("1epoch 소요 시간: {}분".format((end_time-start_time)/60))

for k in range(5):
    step = 1
    train(epochs=30, step=step, loop=k+1)
