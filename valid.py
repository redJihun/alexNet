# For files
import os
import time

# Deep-learning framework
import tensorflow as tf
# import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# Manipulate
import numpy as np
import random

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
IMAGENET_MEAN = np.array([104., 117., 124.], dtype=np.float)

# Data directory
INPUT_ROOT_DIR = './input'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
OUTPUT_ROOT_DIR = './output'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tiny_imagenet')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


########################################################################################################################
# 이미지 로드 및 전처리 + 데이터 증강 함수들

# After init variables, append imagefile's path and label(in number, origin is name of sub-directory).
# Set the path of root dir, and use os.walk(root_dir) for append all images in sub-dir.
def load_imagepaths(path):
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

    return imagepaths, labels


# 256x256으로 이미지 다운샘플링
def resize_images(imgpaths):
    # Read images from disk & resize
    print('start resizing image')
    images = list()
    for img in imgpaths:
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=(256, 256), method=tf.image.ResizeMethod.LANCZOS5)
        images.append(img)
    print('end resizing')
    return images


# RGB jittering
def fancy_pca(images, labels, alpha_std=0.1):
    print('Start Jittering')
    pca_images,pca_labels = images.copy(),labels.copy()
    for img,lbl in zip(images, labels):
        orig_img = img.numpy().copy()
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
        np.random.seed(RANDOM_SEED)
        alpha = np.random.normal(0, alpha_std)
        m2[:, 0] = alpha * eig_vals
        # 3x3 고유벡터 행렬과 3x1 랜덤값*고유값 행렬을 곱해서 3x1 행렬을 얻음(RGB 채널에 가감해줄 값)
        add_vect = np.matrix(m1) * np.matrix(m2)
        # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
        for idx in range(3):
            orig_img[..., idx] += add_vect[idx]
        # 0~255(rgb픽셀값) 범위로 값 재설정
        pca_img = np.clip(orig_img, 0.0, 255.0)
        pca_img = pca_img.astype(np.float)
        pca_images.append(pca_img)
        pca_labels.append(lbl)
    print('End jittering')
    return pca_images, pca_labels


# horizontal reflection
def flip_image(images, labels):
    print('Start flipping')
    flipped_images,flipped_labels = images.copy(),labels.copy()
    for img,lbl in zip(images,labels):
        flipped_image = tf.image.flip_left_right(img)
        flipped_images.append(flipped_image)
        flipped_labels.append(lbl)
    print('End flipping')
    return flipped_images, flipped_labels


# Image cropping
def crop_image(images, labels):
    print('Start cropping')
    cropped_images, cropped_labels = list(), list()
    for img,label in zip(images,labels):
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
    print('End cropping')
    return cropped_images, cropped_labels


# 증강된 데이터를 입력받아 셔플 후 TF 데이터셋으로 리턴
def make_dataset(images, labels):
    print('Start making dataset')
    # Shuffle with seed can keep the data-label pair. Without shuffle, data have same label in range.
    foo = list(zip(images, labels))
    random.Random(RANDOM_SEED).shuffle(foo)
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
    # valid_X, valid_Y = tf.data.Dataset.from_tensor_slices(tensors=valid_X), tf.data.Dataset.from_tensor_slices(tensors=valid_Y)
    # test_X, test_Y = tf.data.Dataset.from_tensor_slices(tensors=test_X), tf.data.Dataset.from_tensor_slices(tensors=test_Y)
    print('End making dataset')
    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y
########################################################################################################################


# @tf.function
def loss(name, x, y, param):
    # inputs = tf.constant(x, name='inputs')
    inputs = x
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
    predict = tf.argmax(logits, 1).numpy()

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(loss)
    target = y
    accuracy = np.sum(predict == target) / len(target)

    print('model\t=\t{}\tloss={}\taccuracy={}'.format(name, loss.numpy(), accuracy))

    return loss


def valid(imgs_path=TRAIN_IMG_DIR, ckpts_path=CHECKPOINT_DIR):
    # 사전에 정의한 load_imagepaths 함수의 매개변수로 이미지를 저장한 파일경로의 루트 디렉토리 지정
    filepaths, labels = load_imagepaths(imgs_path)
    images = resize_images(filepaths)
    images,labels = fancy_pca(images,labels)
    images,labels = crop_image(images,labels)
    images,labels = flip_image(images,labels)
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = make_dataset(images,labels)

    # Trained model loading
    model_paths = list()
    walk = os.walk(ckpts_path).__next__()
    for file in walk[2]:
        model_paths.append(os.path.join(ckpts_path, file))
    # Validation step 에서 최소 loss 기록 모델을 best model로 선정
    min_loss = 99999999
    best_model = dict()
    # 저장된 trained 모델(=trained parameters) 들을 불러온 후, valid set 에서 loss 계산
    for model in model_paths:
        loaded_param = np.load(model, allow_pickle=True)
        loaded_param = {key: loaded_param[key].item() for key in loaded_param}
        print('model : {} // {}'.format(model, loaded_param['arr_0']['b8']))
        current_loss = loss(name=model, x=valid_X, y=valid_Y, param=loaded_param['arr_0'])
        # 저장된 최소 loss보다 작으면 best model 업데이트
        if current_loss < min_loss:
            min_loss = current_loss
            best_model = loaded_param['arr_0'].copy()
    # 최종으로 업데이트된 best model을 저장
    np.savez(os.path.join(OUTPUT_ROOT_DIR, 't_i_best_model'), best_model)
    print("\nBest model : loss={}\n{}".format(min_loss, best_model['b8']))


valid()
