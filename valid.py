import os
import tensorflow as tf
import numpy as np
import random

RANDOM_SEED = 602
# random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Hyper-parameters
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


def loss(name, x, y, param):
    inputs = tf.constant(x, name = 'inputs')

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

    print('model = {}\tloss = {}\taccuracy = {}'.format(name, loss.numpy(), accuracy))

    return loss


train_X, train_Y, valid_X, valid_Y, test_X, test_Y = load_dataset(TRAIN_IMG_DIR)
param_paths = list()
walk = os.walk(CHECKPOINT_DIR).__next__()
for file in walk[2]:
    param_paths.append(os.path.join(CHECKPOINT_DIR, file))

min_loss = 99999999
best_model = dict()
for model in param_paths:
    loaded_param = np.load(model)
    print(loaded_param)
    loss = loss(file, valid_X, valid_Y, param=loaded_param)
    if loss < min_loss:
        min_loss = loss
        best_model = loaded_param

np.savez(os.path.join(OUTPUT_ROOT_DIR, 'best_model'), best_model)
