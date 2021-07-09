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
tf.random.set_seed(RANDOM_SEED)

# Hyper-parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005         # == weight_decay
LR_INIT = 0.01
NUM_CLASSES = 6
IMAGENET_MEAN = np.array([104., 117., 124.], dtype=np.float)

# Data directory
INPUT_ROOT_DIR = './input/task'
VALID_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'valid')
OUTPUT_ROOT_DIR = './output/task'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, '1')

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

    foo = list(zip(imagepaths, labels))
    random.Random(RANDOM_SEED).shuffle(foo)
    imagepaths, labels = zip(*foo)

    return imagepaths, labels


# 256x256으로 이미지 다운샘플링
def resize_images(imgpaths, w, h):
    # Read images from disk & resize
    images = list()
    for img in imgpaths:
        image = cv2.imread(img)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
        images.append(image)
    return images


def minmax(images, min, max):
    scaled_images = list()
    for img in images:
        # R, G, B 채널을 각각 순회하며 계산된 값을 각 픽셀마다 가감
        scaled_img = np.array(img).copy()
        for idx in range(3):
            scaled_img[..., idx] = minmax_scale(img[..., idx], feature_range=(min, max))
        scaled_images.append(scaled_img)

    return scaled_images


# 증강된 데이터를 입력받아 셔플 후 TF 데이터셋으로 리턴
def make_dataset(images, labels):
    # print('Start making dataset')

    # Shuffle with seed can keep the data-label pair. Without shuffle, data have same label in range.
    # foo = list(zip(images, labels))
    # random.Random(RANDOM_SEED).shuffle(foo)
    # images, labels = zip(*foo)

    # Convert to Tensor
    valid_X, valid_Y = tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)

    valid_X, valid_Y = tf.data.Dataset.from_tensor_slices(tensors=valid_X).batch(batch_size=BATCH_SIZE), tf.data.Dataset.from_tensor_slices(tensors=valid_Y).batch(batch_size=BATCH_SIZE)
    # print('End making dataset')
    return valid_X, valid_Y
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

    # layer 7
    l7_fc = tf.nn.bias_add(tf.matmul(l6_relu, param['w7']), param['b7'], name='l7_fc')
    l7_relu = tf.nn.relu(l7_fc, name='l7_relu')

    # layer 8
    logits = tf.nn.bias_add(tf.matmul(l7_relu, param['w8']), param['b8'], name='l8_fc')
    predict = tf.argmax(logits, 1).numpy()

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, depth=NUM_CLASSES), logits=logits)
    loss = tf.reduce_mean(loss)
    target = y
    accuracy = np.sum(predict == target) / len(target)

    # print('model\t=\t{}\tloss={}\taccuracy={}'.format(name, loss.numpy(), accuracy))

    return loss, accuracy


def valid(imgs_path=VALID_IMG_DIR, ckpts_path=CHECKPOINT_DIR):
    # 사전에 정의한 load_imagepaths 함수의 매개변수로 이미지를 저장한 파일경로의 루트 디렉토리 지정
    filepaths, labels = load_imagepaths(imgs_path)

    # Trained model loading
    model_paths = list()
    walk = os.walk(ckpts_path).__next__()
    for file in walk[2]:
        model_paths.append(os.path.join(ckpts_path, file))

    model_paths.sort()

    # Validation step 에서 최소 loss 기록 모델을 best model로 선정
    min_loss = 99999999
    accuracy = 0
    best_model = dict()

    # 저장된 trained 모델(=trained parameters) 들을 불러온 후, valid set 에서 loss 계산
    for model in model_paths:
        loaded_param = np.load(model, allow_pickle=True)
        loaded_param = {key: loaded_param[key].item() for key in loaded_param}
        print('\nmodel : {} // {}'.format(model, loaded_param['arr_0']['b8']))
        losses, accs = list(), list()

        for i in range(int(np.ceil(len(filepaths)/BATCH_SIZE))):
            # 마지막 split은 전체 데이터 개수가 32로 안 나누어 떨어지는 경우 남은 개수만큼만 로드
            if i == int(np.ceil(len(filepaths) / BATCH_SIZE)) - 1:
                fpaths, lbls = filepaths[i * BATCH_SIZE:], list(labels[i * BATCH_SIZE:])
            # 그 외의 split은 32의 배수로 나누어서 로드
            else:
                fpaths, lbls = filepaths[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], list(labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])

            imgs = resize_images(fpaths, 227, 227)
            imgs = minmax(imgs, -1.0, 1.0)
            valid_X, valid_Y = make_dataset(imgs, lbls)

            for batch_X, batch_Y in zip(list(valid_X.as_numpy_iterator()), list(valid_Y.as_numpy_iterator())):
                current_loss, current_acc = loss(name=model, x=batch_X, y=batch_Y, param=loaded_param['arr_0'])
                losses.append(current_loss)
                accs.append(current_acc)

        print('loss: {}\tacc: {}'.format(np.mean(losses), np.mean(accs)))
        # 저장된 최소 loss보다 작으면 best model 업데이트
        if np.mean(losses)-np.mean(accs) < min_loss-accuracy:
            min_loss, accuracy = np.mean(losses), np.mean(accs)
            model_name = model
            best_model = loaded_param['arr_0'].copy()

    try:
        # 최종으로 업데이트된 best model을 저장
        orig_best = np.load(os.path.join(OUTPUT_ROOT_DIR, 'best_model.npz'), allow_pickle=True)
        orig_best = {key: orig_best[key].item() for key in orig_best}
        orig_losses, orig_accs = list(), list()

        for i in range(int(np.ceil(len(filepaths) / BATCH_SIZE))):
            # 마지막 split은 전체 데이터 개수가 BATCH_SIZE로 안 나누어 떨어지는 경우 남은 개수만큼만 로드
            if i == int(np.ceil(len(filepaths) / BATCH_SIZE)) - 1:
                fpaths, lbls = filepaths[i * BATCH_SIZE:], list(labels[i * BATCH_SIZE:])
            # 그 외의 split은 BATCH_SIZE의 배수로 나누어서 로드
            else:
                fpaths, lbls = filepaths[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], list(
                    labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])

            imgs = resize_images(fpaths, 227, 227)
            imgs = minmax(imgs, -1.0, 1.0)
            valid_X, valid_Y = make_dataset(imgs, lbls)

            for batch_X, batch_Y in zip(list(valid_X.as_numpy_iterator()), list(valid_Y.as_numpy_iterator())):
                orig_loss, orig_acc = loss(name='orig_best', x=batch_X, y=batch_Y, param=orig_best['arr_0'])
                orig_losses.append(orig_loss)
                orig_accs.append(orig_acc)

        # 저장된 최소 loss보다 작으면 best model 업데이트
        if np.mean(orig_losses) - np.mean(orig_accs) > min_loss - accuracy:
            np.savez(os.path.join(OUTPUT_ROOT_DIR, 'best_model'), best_model)
            print("\nBest model : {}\nloss={}\taccuracy={}\n{}".format(model_name, min_loss, accuracy, best_model['b8']))
        else:
            print("\nBest model : orig_best\nloss={}\taccuracy={}\n{}".format(np.mean(orig_losses), np.mean(orig_accs), orig_best['arr_0']['b8']))

    except:
        np.savez(os.path.join(OUTPUT_ROOT_DIR, 'best_model'), best_model)
        print("\nBest model : {}\nloss={}\taccuracy={}\n{}".format(model_name, min_loss, accuracy, best_model['b8']))


valid()
