""" Deep Convolutional Generative Adversarial Network (DCGAN).

Using deep convolutional generative adversarial networks (DCGAN) to generate
digit images from a noise distribution.

References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.

Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/home/cadlab/ECE283_GAN/ArchData224/train')
#parser.add_argument('--val_dir', default='/home/cadlab/ECE283_GAN/ArchData224/val')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=15, type=int)
parser.add_argument('--num_epochs2', default=15, type=int)
parser.add_argument('--learning_rate2', default=5e-4, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    label = labels[0]
    #print(labels.shape)
    #print(labels)
    files_and_labels = []
    #for label in labels:
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



# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 10000
batch_size = 8

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points

# Training Params

lr_generator = 0.002
lr_discriminator = 0.002

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
# A boolean to indicate batch normalization if it is training or inference time
is_training = tf.placeholder(tf.bool)


#LeakyReLU activation
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)


# Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=7 * 7 * 128)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2, padding='same')
        # Deconvolution, image shape: (batch, 56, 56, 64)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 32, 4, strides=2, padding='same')
        # Deconvolution, image shape: (batch, 112, 112, 32)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 16, 4, strides=2, padding='same')
        # Deconvolution, image shape: (batch, 224, 224, 16)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 3, 2, strides=2, padding='same')
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.tanh(x)
        return x


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x


# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 224.0
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
    #flip_image = tf.image.random_flip_left_right(crop_image)  # (4)

    #means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    #centered_image = flip_image #- means  # (5)

    return crop_image, label


# Data preprocess
args = parser.parse_args()
train_filenames, train_labels = list_images(args.train_dir)
# val_filenames, val_labels = list_images(args.val_dir)
num_classes = len(set(train_labels))

# ----------------------------------------------------------------------
# DATASET CREATION using tf.contrib.data.Dataset
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

# The tf.contrib.data.Dataset framework uses queues in the background to feed in
# data to the model.
# We initialize the dataset with a list of filenames and labels, and then apply
# the preprocessing functions described above.
# Behind the scenes, queues will load the filenames, preprocess them with multiple
# threads and apply the preprocessing in parallel, and then batch the data

# Training dataset

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


train_init_op = iterator.make_initializer(batched_train_dataset)
# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
#real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Loss (Labels for real images: 1, for fake images: 0)
# Discriminator Loss for real and fake samples
disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
# Sum both loss
disc_loss = disc_loss_real + disc_loss_fake
# Generator Loss (The generator tries to fool the discriminator, thus labels are 1)
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)


# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
# TensorFlow UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev
gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
# `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)
with tf.control_dependencies(gen_update_ops):
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    sess.run(train_init_op)
    #saver.restore(sess, "/home/cadlab/PycharmProjects/GAN_Architec/May23_2000model.ckpt")
    for i in range(1, num_steps+1):
        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x, _ = iterator.get_next()
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 3])
        # Rescale to [-1, 1], the input range of the discriminator
        batch_x = batch_x * 2. - 1.
        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, _ = mnist.train.next_batch(batch_size)
        #batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the generator are real images,
        # the other half are fake images (coming from the generator).
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([batch_size])

        # Training
        #feed_dict = {real_image_input: batch_x, noise_input: z,
        #feed_dict = {noise_input: z,
        #            disc_target: batch_disc_y, gen_target: batch_gen_y,is_training:True}
        #_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
        #                       feed_dict=feed_dict)
        # Discriminator Training
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, dl = sess.run([train_disc, disc_loss],
                         feed_dict={noise_input: z, is_training: True})

        # Generator Training
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, is_training: True})

        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
    save_path = saver.save(sess, "/home/cadlab/PycharmProjects/GAN_Architec/May23_2000_newmodel.ckpt")
    print("Model saved in path: %s" % save_path)
    #saver.restore(sess, "/home/cadlab/PycharmProjects/GAN_Architec/May23_2000model.ckpt")
    # Generate images from noise, using the generator network.
    f, a = plt.subplots(2, 2)#, figsize=(10, 4))
    for i in range(2):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[2, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z, is_training: False})
        for j in range(2):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 1, axis=2),
                             newshape=(224, 224, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()
