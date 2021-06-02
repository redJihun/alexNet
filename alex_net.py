# For files
import os

# Deep-learning framework
import tensorflow as tf
import tensorflow_datasets as tfds

# Manipulate
import numpy as np

# Hyper-parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005         # == weight_decay
LR_INIT = 0.01            # == weight_init
IMAGE_DIM = 227
NUM_CLASSES = 200

# Initialized the weights in each layer from a zero-mean Gaussian distribution(standard deviation=0.01)
# initialized bias with constant 1, 2/4/5 conv layers, fully-connected hidden layers
# else layers initialized with constant 0

# Data directory
INPUT_ROOT_DIR = './'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
# VAL_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'valid')
# TEST_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'test')
OUTPUT_ROOT_DIR = '/home/alexnet/output'
LOG_DIR = os.path.join(OUTPUT_ROOT_DIR, 'tblogs')
CHECKPOINT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'models')

# Make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# @todo Load dataset
# builder = tfds.ImageFolder(INPUT_ROOT_DIR)
# datasets = builder.as_dataset(split='train')
# tfds.load(datasets, split='train[:80%]')
# print(datasets)
imagepaths, labels = list(), list()
label = 0
classes = sorted(os.walk(TRAIN_IMG_DIR).__next__()[1])
for c in classes:
    c_dir = os.path.join(TRAIN_IMG_DIR, c)
    walk = os.walk(c_dir).__next__()
    # Add each image to the training set
    for sample in walk[2]:
        # Only keeps jpeg images
        # if sample.endswith('.jpg') or sample.endswith('.jpeg'):
        imagepaths.append(os.path.join(c_dir, sample))
        labels.append(label)
    label += 1

# Convert to Tensor
imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)
# Build Tf Queue, shuffle data
image = tf.data.Dataset.from_tensor_slices(tensors=imagepaths).shuffle(1024, seed=602)
label = tf.data.Dataset.from_tensor_slices(tensors=labels).shuffle(1024, seed=602)

# Read images from disk
images = list()
for img in image:
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=(256, 256))
    img = tf.image.crop_and_resize(img, crop_size=(227, 227), boxes=[900, 4], box_indices=)

# Resize images
# image = tf.image.resize(image, size=(256,256))
# train_image, train_label = image[:80000], label[:80000]
# valid_image, valid_label = image[80000:90000], label[80000:90000]
# test_image, test_label = image[-10000:], label[-10000:]
# image = tf.image.crop_and_resize(image, crop_size=(227,227), boxes=[900, 4])

# train_dataset, valid_dataset, test_dataset =
# print(count=[x for x, y in enumerate(datasets)][-1]+1)
# print(datasets.take(1)['image'])

#----------------------------------------------------------------------------------------------------------------------
# Define conv, fc, max_pool, lrn, dropout method
def conv(input, weight, bias, strides, name, padding='VALID'):
    """
    Apply the convolution to input with filters(filter == weight).
    Then add bias and apply the activation function(In AlexNet they have only ReLU function).
    :param input:
    :param weight:
    :param bias:
    :param strides:
    :param name:
    :param padding:
    :return:
    """
    # Do convolution
    convolve = tf.nn.conv2d(input, weight, strides=[strides], padding=padding)

    # Add bias
    bias = tf.reshape(tf.nn.bias_add(convolve, bias), tf.shape(convolve))

    # Apply activation
    relu = tf.nn.relu(bias, name=name)

    return relu


def fc(input, weight, bias, name, activation='relu'):
    """
    Matrix input multiply with weights and add bias.
    Then apply the activation function.
    :param input:
    :param weight:
    :param bias: The bias of this layer. If output is first Fully-connected layer, bias is parameters['b6']
    :param name: The name of output.(e.g., 'fc1', 'fc2')
    :param activation: Name of activation function.(e.g., 'relu', 'softmax' etc.)
    :return: tf.tensor(applied activation)
    """
    foo = tf.nn.bias_add(tf.matmul(input, weight), bias)

    if activation == 'relu':
        act = tf.nn.relu(foo, name=name)
    else:
        act = tf.nn.softmax(foo, name=name)

    return act


def max_pool(input, name, ksize=3, strides=[1, 2, 2, 1], padding='VALID'):
    """
    Apply the max_pooling. All max_pooling layer have same ksize and strides.
    So just input input, name and sometimes padding('VALID' or 'SAME').
    :param input:
    :param name:
    :param ksize:
    :param strides:
    :param padding:
    :return:
    """
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)


def lrn(input, name, radius=5, alpha=10e-4, beta=0.75, bias=2.0):
    """
    All local_response_normalization layers have same hyper-parameters.
    So just input input and name.
    :param input:
    :param name:
    :param radius:
    :param alpha:
    :param beta:
    :param bias:
    :return:
    """
    return tf.nn.local_response_normalization(input=input, depth_radius=radius,
                                              alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(input, keep_prob=0.5):
    """
    All dropout layers have same rate. So give them default value.
    :param input:
    :param keep_prob:
    :return:
    """
    return tf.nn.dropout(input, rate=keep_prob)

#----------------------------------------------------------------------------------------------------------------------


# Make CNN model
def alexnet(X, parameters):
    # Layer 1 : Convolution -> LRN -> Max pooling
    l1_conv = conv(input=X, weight=parameters['w1'], bias=parameters['b1'], strides=[1, 4, 4, 1], name='l1_conv')
    l1_norm = lrn(input=l1_conv, name='l1_norm')
    l1_pool = max_pool(input=l1_norm, name='l1_pool')

    # Layer 2 : Convolution -> LRN -> Max pooling
    l2_conv = conv(input=l1_pool, weight=parameters['w2'], bias=parameters['b2'], strides=[1, 1, 1, 1], name='l2_conv')
    l2_norm = lrn(input=l2_conv, name='l2_norm')
    l2_pool = max_pool(input=l2_norm, name='l2_pool')

    # Layer 3 : Convolution
    l3_conv = conv(input=l2_pool, weight=parameters['w3'], bias=parameters['b3'], strides=[1, 1, 1, 1], name='l3_conv')

    # Layer 4 : Convolution
    l4_conv = conv(input=l3_conv, weight=parameters['w4'], bias=parameters['b4'], strides=[1, 1, 1, 1], name='l4_conv', padding='SAME')

    # Layer 5 : Convolution -> Max pooling
    l5_conv = conv(input=l4_conv, weight=parameters['w5'], bias=parameters['b5'], strides=[1, 1, 1, 1], name='l5_conv', padding='SAME')
    l5_pool = max_pool(input=l5_conv, name='l5_pool')

    # Layer 6 : Flatten -> Fully connected -> Dropout
    l6_flattened = tf.reshape(l5_pool, [-1, tf.shape(parameters['w6'])[0]])
    l6_fc = fc(input=l6_flattened, weight=parameters['w6'], bias=parameters['b6'], name='l6_fc')
    l6_dropout = dropout(input=l6_fc)

    # Layer 7 : Fully connected -> Dropout
    l7_fc = fc(input=l6_dropout, weight=parameters['w7'], bias=parameters['b7'], name='l7_fc')
    l7_dropout = dropout(input=l7_fc)

    # Layer 8 : Fully connected(with softmax)   # Output layer
    l8_fc = fc(input=l7_dropout, weight=parameters['w8'], bias=parameters['b8'], name='l8_fc')

    return l8_fc





# @todo image down sampling - 짧은 면 256픽셀 + 긴면 같은 비율로 줄임, 긴 면의 가운데 256픽셀 자름 -> 256x256 이미지


# @todo image preprocessing - 각 픽셀에서 이미지의 픽셀 값 평균을 빼줌(픽셀 평균을 0으로 만듦)


# @todo Data augmentation - crop, RGB(pca)

# Initialize variables
# weight init with Gaussian distribution(mean=0.0 & standard_deviation=0.01)
# bias init 1/3/8 = 0, 2/4/5/6/7 = 1
tf.random.set_seed(602)
parameters = {
    'w1': tf.Variable(tf.random.normal(shape=[11, 11, 3, 96], mean=0.0, stddev=0.01, dtype=tf.float32), name='w1', trainable=True),
    'b1': tf.Variable(tf.zeros(shape=[96], name='b1'), trainable=True),

    'w2': tf.Variable(tf.random.normal(shape=[5, 5, 96, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w2', trainable=True),
    'b2': tf.Variable(tf.ones(shape=[256], name='b2'), trainable=True),

    'w3': tf.Variable(tf.random.normal(shape=[3, 3, 256, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w3', trainable=True),
    'b3': tf.Variable(tf.zeros(shape=[384], name='b3'), trainable=True),

    'w4': tf.Variable(tf.random.normal(shape=[3, 3, 384, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w4', trainable=True),
    'b4': tf.Variable(tf.ones(shape=[384], name='b4'), trainable=True),

    'w5': tf.Variable(tf.random.normal(shape=[3, 3, 384, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w5', trainable=True),
    'b5': tf.Variable(tf.ones(shape=[256], name='b5'), trainable=True),

    'w6': tf.Variable(tf.random.normal(shape=[6*6*256, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w6', trainable=True),
    'b6': tf.Variable(tf.ones(shape=[4096], name='b6'), trainable=True),

    'w7': tf.Variable(tf.random.normal(shape=[4096, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w7', trainable=True),
    'b7': tf.Variable(tf.ones(shape=[4096], name='b7'), trainable=True),

    'w8': tf.Variable(tf.random.normal(shape=[4096, NUM_CLASSES], mean=0.0, stddev=0.01, dtype=tf.float32), name='w8', trainable=True),
    'b8': tf.Variable(tf.zeros(shape=[NUM_CLASSES], name='b8'), trainable=True),
}


# @todo Do training
# Launch the session
# with tf.Session() as sess:
#     tf.initialize_all_variables().run()

    # for epoch in range(NUM_EPOCHS):

# @todo Do validation check & model save
