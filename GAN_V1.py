import argparse
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Model parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/home/cadlab/ECE283_GAN/ArchData224/train')
parser.add_argument('--val_dir', default='/home/cadlab/ECE283_GAN/ArchData224/val')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=15, type=int)
parser.add_argument('--num_epochs2', default=15, type=int)
parser.add_argument('--learning_rate2', default=5e-4, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]

# Training Params
num_steps = 70000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784 # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100 # Noise data points


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels



# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}

# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Discriminator
def discriminator(images):
    C = 3
    filterSize = 4
    filterNum = 32
    filterStride = 1
    poolSize = 2
    poolStride = 2
    hiddenSize = 1024
    classNum = 6
    # print('*********************************** images ******************************************')
    # print(images.shape)
    # print('*************************************************************************************')
    scoreConvReLU1 = tf.layers.conv2d(inputs=images, filters=filterNum, kernel_size=filterSize, padding='SAME',
                                      activation=tf.nn.relu)
    # print('*********************************** scoreConvReLU1 ******************************************')
    # print(scoreConvReLU1.shape)
    # print('*************************************************************************************')
    scoreConvReLU2 = tf.layers.conv2d(inputs=scoreConvReLU1, filters=filterNum, kernel_size=filterSize, padding='SAME',
                                      activation=tf.nn.relu)
    # print('*********************************** scoreConvReLU2 ******************************************')
    # print(scoreConvReLU2.shape)
    # print('*************************************************************************************')
    scoreConvReLUPool2 = tf.layers.max_pooling2d(inputs=scoreConvReLU2, pool_size=poolSize, strides=poolStride,
                                                 padding='SAME', data_format='channels_last')
    # print('*********************************** scoreConvReLUPool2 ******************************************')
    # print(scoreConvReLUPool2.shape)
    # print('*************************************************************************************')
    scoreConvReLU3 = tf.layers.conv2d(inputs=scoreConvReLUPool2, filters=filterNum, kernel_size=filterSize,
                                      padding='SAME', activation=tf.nn.relu)
    # print('*********************************** scoreConvReLU3 ******************************************')
    # print(scoreConvReLU3.shape)
    # print('*************************************************************************************')
    scoreConvReLU4 = tf.layers.conv2d(inputs=scoreConvReLU3, filters=filterNum, kernel_size=[filterSize, filterSize],
                                      padding='SAME', activation=tf.nn.relu)
    # print('*********************************** scoreConvReLU4 ******************************************')
    # print(scoreConvReLU4.shape)
    # print('*************************************************************************************')
    scoreConvReLUPool4 = tf.layers.max_pooling2d(scoreConvReLU4, pool_size=[poolSize, poolSize], strides=poolStride,
                                                 padding='SAME', data_format='channels_last')
    # print('*********************************** scoreConvReLUPool4 ******************************************')
    # print(scoreConvReLUPool4.shape)
    # print('*************************************************************************************')
    scoreConvReLU5 = tf.layers.conv2d(inputs=scoreConvReLUPool4, filters=filterNum,
                                      kernel_size=[filterSize, filterSize], padding='SAME', activation=tf.nn.relu)
    # print('*********************************** scoreConvReLU5 ******************************************')
    # print(scoreConvReLU5.shape)
    # print('*************************************************************************************')
    scoreConvReLU6 = tf.layers.conv2d(inputs=scoreConvReLU5, filters=filterNum, kernel_size=[filterSize, filterSize],
                                      padding='SAME', activation=tf.nn.relu)
    # print('*********************************** scoreConvReLU6 ******************************************')
    # print(scoreConvReLU6.shape)
    # print('*************************************************************************************')
    scoreConvReLUPool6 = tf.layers.max_pooling2d(scoreConvReLU6, pool_size=[poolSize, poolSize], strides=poolStride,
                                                 padding='SAME', data_format='channels_last')
    # print('*********************************** scoreConvReLUPool6 ******************************************')
    # print(scoreConvReLUPool6.shape)
    # print('*************************************************************************************')
    scoreAvgPool = tf.layers.average_pooling2d(scoreConvReLUPool6, (poolSize, poolSize), (poolStride, poolStride),
                                               padding='VALID', data_format='channels_last')
    # print('*******************************************scoreAvgPool**********************************')
    # print(scoreAvgPool.shape)
    # print('*************************************************************************************')

    scoreAvgPool_Flat = tf.reshape(scoreAvgPool, [-1, 14 * 14 * 32])
    # print('*******************************************scoreAvgPool_Flat**********************************')
    # print(scoreAvgPool_Flat.shape)
    # print('*************************************************************************************')
    scoreBN = tf.layers.batch_normalization(scoreAvgPool_Flat)
    # print('*******************************************scoreBN**********************************')
    # print(scoreBN.shape)
    # print('*************************************************************************************')
    scoreAff1 = tf.layers.dense(inputs=scoreBN, units=hiddenSize, activation=tf.nn.relu)
    # print('*******************************************scoreAff1**********************************')
    # print(scoreAff1.shape)
    # print('*************************************************************************************')
    scoreAff2 = tf.layers.dense(inputs=scoreAff1, units=hiddenSize, activation=tf.nn.relu)
    # print('*******************************************scoreAff2**********************************')
    # print(scoreAff2.shape)
    # print('*************************************************************************************')
    scoreAff3 = tf.layers.dense(inputs=scoreAff2, units=hiddenSize, activation=tf.nn.relu)
    # print('*******************************************scoreAff3**********************************')
    # print(scoreAff3.shape)
    # print('*************************************************************************************')
    scoreAff4 = tf.layers.dense(inputs=scoreAff3, units=hiddenSize, activation=tf.nn.relu)
    # print('*******************************************scoreAff4**********************************')
    # print(scoreAff4.shape)
    # print('*************************************************************************************')
    scoreAff5 = tf.layers.dense(inputs=scoreAff4, units=hiddenSize, activation=tf.nn.relu)
    # print('*******************************************scoreAff5**********************************')
    # print(scoreAff5.shape)
    # print('*************************************************************************************')
    logits = tf.layers.dense(inputs=scoreAff5, units=classNum)

    #hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    #hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    #hidden_layer = tf.nn.relu(hidden_layer)
    #out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    #out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(logits)
    return out_layer


# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label


# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess(image, label):
    crop_image = tf.random_crop(image, [224, 224, 3])  # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)  # (4)

    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = flip_image - means  # (5)

    return centered_image, label


# Input Iterator
# Training dataset
args = parser.parse_args()
train_filenames, train_labels = list_images(args.train_dir)
val_filenames, val_labels = list_images(args.val_dir)
num_classes = len(set(train_labels))


train_filenames = tf.constant(train_filenames)
train_labels = tf.constant(train_labels)
train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.map(_parse_function,
    num_threads=args.num_workers, output_buffer_size=args.batch_size)
train_dataset = train_dataset.map(training_preprocess,
    num_threads=args.num_workers, output_buffer_size=args.batch_size)
train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
batched_train_dataset = train_dataset.batch(args.batch_size)

# Now we define an iterator that can operator on either dataset.
# The iterator can be reinitialized by calling:
#     - sess.run(train_init_op) for 1 epoch on the training set
#     - sess.run(val_init_op)   for 1 epoch on the valiation set
# Once this is done, we don't need to feed any value for images and labels
# as they are automatically pulled out from the iterator queues.

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `train_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                   batched_train_dataset.output_shapes)
images, labels = iterator.get_next()

train_init_op = iterator.make_initializer(batched_train_dataset)


# Build Networks
# Network Inputs
gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
#disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

# Build Generator Network
gen_sample = generator(gen_input)
#gen_sample = generator(images)
# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(images)
disc_fake = discriminator(gen_sample)

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
            biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

# Training
for i in range(1, num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    # batch_x, _ = mnist.train.next_batch(batch_size)
    sess.run(train_init_op)
    # Generate noise to feed to the generator
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

    # Train
    # feed_dict = {disc_input: batch_x, gen_input: z}
    feed_dict = {gen_input: z}
    _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                            feed_dict=feed_dict)
    if i % 2000 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))


# Testing
# Generate images from noise, using the generator network.
n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # Generate image from noise.
    g = sess.run(gen_sample, feed_dict={gen_input: z})
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()