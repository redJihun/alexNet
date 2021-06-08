# For files
import os

# Deep-learning framework
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# Manipulate
import numpy as np
import random

RANDOM_SEED = 602
random.seed(RANDOM_SEED)
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
imagepaths, labels = list(), list()
label = 0
classes = sorted(os.walk(TRAIN_IMG_DIR).__next__()[1])
for c in classes:
    c_dir = os.path.join(TRAIN_IMG_DIR, c)
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
random.shuffle(images)
random.shuffle(labels)

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
########################################################################################################################


########################################################################################################################
# Define conv, fc, max_pool, lrn, dropout method
def conv(input, weight, bias, strides, name, padding='VALID'):
    """
    Apply the convolution to input with filters(filter == weight).
    Then add bias and apply the activation function(In AlexNet they have only ReLU function).
    :param input: (Tensor)Placeholder for input tensor.
    :param weight: (Tensor)Placeholder for input weight.
    :param bias: Placeholder for input bias.
    :param strides: (Integer)Strides of the convolution layer.
    :param name: (String)The name of output.
    :param padding: (String)Nome of applied padding style.'VALID' of 'SAME'.
    :return: Return the applied convolution layer that with input parameters.
    """
    # Do convolution
    convolve = tf.nn.conv2d(input, weight, strides=strides, padding=padding)

    # Add bias
    bias = tf.reshape(tf.nn.bias_add(convolve, bias), tf.shape(convolve))

    # Apply activation
    relu = tf.nn.relu(bias, name=name)

    return relu


def fc(input, weight, bias, name, activation='relu'):
    """
    Matrix input multiply with weights and add bias.
    Then apply the activation function.
    :param input: Placeholder for input tensor.
    :param weight: Placeholder for input weight.
    :param bias: Placeholder for input bias. If output is first Fully-connected layer, bias is parameters['b6']
    :param name: The name of output.(e.g., 'fc1', 'fc2')
    :param activation: Name of activation function.(e.g., 'relu', 'softmax' etc.)
    :return: tf.tensor(applied activation)
    """
    foo = tf.nn.bias_add(tf.matmul(input, weight), bias)

    if activation == 'relu':
        act = tf.nn.relu(foo, name=name)
    else:
        act = tf.nn.softmax(foo, name=name, axis=1)

    return act


def max_pool(input, name, ksize=3, strides=2, padding='VALID'):
    """
    Apply the max_pooling. All max_pooling layer have same ksize and strides.
    :param input: (Tensor)Placeholder for input tensor.
    :param name: (String)Name for return layer.
    :param ksize: (Integer)Kernel size of the pooling layer.
    :param strides: (Integer)Strides of the pooling layer.
    :param padding: (String)Padding of the pooling layer. 'VALID' or 'SAME'
    :return: Applied max-pooling layer that used input parameters.
    """
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)


def lrn(input, name, radius=5, alpha=10e-4, beta=0.75, bias=2.0):
    """
    All local_response_normalization layers have same hyper-parameters.
    So just input input and name.
    :param input: Placeholder for input tensor.
    :param name: Name for return.
    :param radius: Depth that use in normalization function.
    :param alpha: Hyperparameter of the lrn function.
    :param beta: Hyperparameter of the lrn function.
    :param bias: Bias of the lrn function.
    :return: LRN layer that applied input parameters.
    """
    return tf.nn.local_response_normalization(input=input, depth_radius=radius,
                                              alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(input, keep_prob=0.5):
    """
    All dropout layers have same rate. So give them default value.
    :param input: Placeholder for input tensor.
    :param keep_prob: Probability of the dropout,
    :return: TF's dropout method that apply the input parameters.
    """
    return tf.nn.dropout(input, rate=keep_prob)


########################################################################################################################
# Make CNN model
class Alexnet(object):
    def init_grads(p):
        """
        Initialize the parameters of model as a python dictionary.
            - keys: 'dw1', 'db1', ... , 'db8'
            - values: Numpy arrays.
        :param param: Python dictionary that contain the parameters.
        :return: Initialized python dictionary.
        """
        num = int(len(p) / 2)
        g = {}

        # Initialize
        for i in range(num):
            g['dw' + str(i + 1)] = tf.Variable(tf.zeros(shape=p['w' + str(i + 1)].shape), trainable=True,
                                               name=('dw' + str(i + 1)))
            g['db' + str(i + 1)] = tf.Variable(tf.zeros(shape=p['b' + str(i + 1)].shape), trainable=True,
                                               name=('db' + str(i + 1)))

        return g

    def __init__(self, weights_path='DEFAULT'):
        """
        Initialize the model's variable
        :param X: Placeholder for the input tensor.
        :param weights_path: The path of pretrained weight file.
        """
        # Parse input arguments into class variables.
        self.num_classes = NUM_CLASSES
        self.param = {
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
        self.grad = self.init_grads(self.param)

        if weights_path == 'DEFAULT':
            self.weights_path = 'alexnet_pretrained.npy'
        else:
            self.weights_path = weights_path

    def run(self, x):
        # Layer 1 : Convolution -> LRN -> Max pooling
        l1_conv = conv(input=x, weight=self.param['w1'], bias=self.param['b1'], strides=4, name='l1_conv')
        l1_norm = lrn(input=l1_conv, name='l1_norm')
        l1_pool = max_pool(input=l1_norm, name='l1_pool')

        # Layer 2 : Convolution -> LRN -> Max pooling
        l2_conv = conv(input=l1_pool, weight=self.param['w2'], bias=self.param['b2'], strides=1, name='l2_conv', padding='SAME')
        l2_norm = lrn(input=l2_conv, name='l2_norm')
        l2_pool = max_pool(input=l2_norm, name='l2_pool')

        # Layer 3 : Convolution
        l3_conv = conv(input=l2_pool, weight=self.param['w3'], bias=self.param['b3'], strides=1, name='l3_conv', padding='SAME')

        # Layer 4 : Convolution
        l4_conv = conv(input=l3_conv, weight=self.param['w4'], bias=self.param['b4'], strides=1, name='l4_conv', padding='SAME')

        # Layer 5 : Convolution -> Max pooling
        l5_conv = conv(input=l4_conv, weight=self.param['w5'], bias=self.param['b5'], strides=1, name='l5_conv', padding='SAME')
        l5_pool = max_pool(input=l5_conv, name='l5_pool')

        # Layer 6 : Flatten -> Fully connected -> Dropout
        l6_flattened = tf.reshape(l5_pool, [-1, tf.shape(self.param['w6'])[0]])
        l6_fc = fc(input=l6_flattened, weight=self.param['w6'], bias=self.param['b6'], name='l6_fc')
        l6_dropout = dropout(input=l6_fc)

        # Layer 7 : Fully connected -> Dropout
        l7_fc = fc(input=l6_dropout, weight=self.param['w7'], bias=self.param['b7'], name='l7_fc')
        l7_dropout = dropout(input=l7_fc)

        # Layer 8 : Fully connected(with softmax)   # Output layer
        l8_fc = fc(input=l7_dropout, weight=self.param['w8'], bias=self.param['b8'], name='l8_fc', activation='softmax')

        return l8_fc

    def load_weights(self):
        """

        :return:
        """
########################################################################################################################


# Initialize variables
# weight init with Gaussian distribution(mean=0.0 & standard_deviation=0.01)
# bias init 1/3/8 = 0, 2/4/5/6/7 = 1
# def init_grads(param):
#     """
#     Initialize the parameters of model as a python dictionary.
#         - keys: 'dw1', 'db1', ... , 'db8'
#         - values: Numpy arrays.
#     :param param: Python dictionary that contain the parameters.
#     :return: Initialized python dictionary.
#     """
#     num = int(len(param)/2)
#     g = {}
#
#     # Initialize
#     for i in range(num):
#         g['dw'+str(i+1)] = tf.Variable(tf.zeros(shape=param['w'+str(i+1)].shape), trainable=True, name=('dw'+str(i+1)))
#         g['db'+str(i+1)] = tf.Variable(tf.zeros(shape=param['b'+str(i+1)].shape), trainable=True, name=('db'+str(i+1)))
#
#     return g


# parameters = {
#     'w1': tf.Variable(tf.random.normal(shape=[11, 11, 3, 96], mean=0.0, stddev=0.01, dtype=tf.float32), name='w1', trainable=True),
#     'b1': tf.Variable(tf.zeros(shape=[96]), trainable=True, name='b1'),
#
#     'w2': tf.Variable(tf.random.normal(shape=[5, 5, 96, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w2', trainable=True),
#     'b2': tf.Variable(tf.ones(shape=[256]), trainable=True, name='b2'),
#
#     'w3': tf.Variable(tf.random.normal(shape=[3, 3, 256, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w3', trainable=True),
#     'b3': tf.Variable(tf.zeros(shape=[384]), trainable=True, name='b3'),
#
#     'w4': tf.Variable(tf.random.normal(shape=[3, 3, 384, 384], mean=0.0, stddev=0.01, dtype=tf.float32), name='w4', trainable=True),
#     'b4': tf.Variable(tf.ones(shape=[384]), trainable=True, name='b4'),
#
#     'w5': tf.Variable(tf.random.normal(shape=[3, 3, 384, 256], mean=0.0, stddev=0.01, dtype=tf.float32), name='w5', trainable=True),
#     'b5': tf.Variable(tf.ones(shape=[256]), trainable=True, name='b5'),
#
#     'w6': tf.Variable(tf.random.normal(shape=[6*6*256, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w6', trainable=True),
#     'b6': tf.Variable(tf.ones(shape=[4096]), trainable=True, name='b6'),
#
#     'w7': tf.Variable(tf.random.normal(shape=[4096, 4096], mean=0.0, stddev=0.01, dtype=tf.float32), name='w7', trainable=True),
#     'b7': tf.Variable(tf.ones(shape=[4096]), trainable=True, name='b7'),
#
#     'w8': tf.Variable(tf.random.normal(shape=[4096, NUM_CLASSES], mean=0.0, stddev=0.01, dtype=tf.float32), name='w8', trainable=True),
#     'b8': tf.Variable(tf.zeros(shape=[NUM_CLASSES]), trainable=True, name='b8'),
# }

# gradients = init_grads(parameters)
########################################################################################################################


# @todo image down sampling - 짧은 면 256픽셀 + 긴면 같은 비율로 줄임, 긴 면의 가운데 256픽셀 자름 -> 256x256 이미지


# @todo image preprocessing - 각 픽셀에서 이미지의 픽셀 값 평균을 빼줌(픽셀 평균을 0으로 만듦)


# @todo Data augmentation - crop, RGB(pca)
# model = alexnet(X, parameters)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))
# optimizer = tf.optimizers.SGD(learning_rate=LR_INIT, momentum=MOMENTUM, weight_decay=LR_DECAY).minimize(cost)
# prediction = tf.argmax(model, 1)


########################################################################################################################
# @todo Do training
# Define loss function
def cost(target_y, predicted_y):
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_y, logits=predicted_y))
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# # Create optimizer and apply gradient descent to the trainable variables
optimizer = tfa.optimizers.SGDW(momentum=MOMENTUM, learning_rate=LR_INIT, weight_decay=LR_DECAY, name='optimizer')


# Define training loop
def train(input_x, input_y):
    print('start train')
    batch_index = 1
    model = Alexnet()
    for batch_x, batch_y in zip(list(input_x.as_numpy_iterator()), list(input_y.as_numpy_iterator())):

        # Loop in number of layer(=number of weights, number of bias)
        with tf.GradientTape() as tape:
            # Trainable variables are tracked by GradientTape
            # tape.watch(model.tra)
            predictions = tf.argmax(model.run(x=batch_x), axis=1)
            current_loss = cost(batch_y, predictions)
            print('current_loss {}'.format(current_loss))

        # Get gradients
        grad = tape.gradient(current_loss, param)
        # [foo, bar] = tape.gradient(y, [param['w1'], param['b1']])
        # optimizer.apply_gradients(zip(foo, param['w1']))
        # optimizer.apply_gradients(zip(bar, param['b1']))
        # for i in range(1, 9):
        #     optimizer.apply_gradients((grad['dw' + str(i)], param['w' + str(i)]))
        #     optimizer.apply_gradients((grad['db' + str(i)], param['b' + str(i)]))

        print('Batch: {} Loss: {}'.format(batch_index, current_loss))
        # for p in parameters.keys():
        #     param[p] = param[p] - param[p]*LR_INIT
        batch_index += 1

    return param, grad, current_loss


for epoch in range(NUM_EPOCHS):
    print('start epoch')
    params, grads, loss = train(train_X, train_Y)

    print('Epoch: {} Loss: {}'.format(epoch, loss))

# Launch the session
# with tf.Session() as sess:
    # tf.initialize_all_variables().run()
    # sess.run(tf.global_variables_initializer())
#
#     for epoch in range(NUM_EPOCHS):
#         sess.run(optimizer, feed_dict={X: train_ds, Y: labels})
#
#     test_indices = np.arange(len())
#     np.random.shuffle(test_indices)
#     test_indices = test_indices[0:test_size]
#     print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
#                      sess.run(predict_op, feed_dict={X: teX[test_indices],
#                                                      Y: teY[test_indices]})))
########################################################################################################################


# @todo Do validation check & model save
