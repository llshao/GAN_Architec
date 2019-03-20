from __future__ import division, print_function, absolute_import
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


### log_path
logs_path='/home/cadlab/PycharmProjects/GAN_Architec/logs/'
# Training Params
DEBUG = True
num_steps = 10000
batch_size = 128
lr_generator = 0.002
lr_discriminator = 0.002

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
noise_dim = 100 # Noise data points
Pic_Len   = 28 # Pic length
Pic_Wid   = 28 # Pic width
Pic_Cha   = 3  # Pic Channel
# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# A boolean to indicate batch normalization if it is training or inference time
is_training = tf.placeholder(tf.bool)


##############################################################
#################From Vanilla#################################

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/home/cadlab/ECE283_GAN/ArchData224/train')
parser.add_argument('--batch_size', default=batch_size, type=int)
parser.add_argument('--num_workers', default=4, type=int)

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



# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=Pic_Cha)  # (1)
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
    crop_image = tf.random_crop(image, [Pic_Len, Pic_Wid, Pic_Cha])  # (3)
    #flip_image = tf.image.random_flip_left_right(crop_image)  # (4)

    #means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    #centered_image = flip_image #- means  # (5)


    return crop_image, label



#######################################################################
###############End of Vanilla#########################################
#LeakyReLU activation
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)


# Generator Network
# Input: Noise, Output: Image
# Note that batch normalization has different behavior at training and inference time,
# we then use a placeholder to indicates the layer if we are training or not.
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=7 * 7 * 128)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 3, 5, strides=2, padding='same')
        # Apply tanh for better stability - clip values to [-1, 1].
        x = tf.nn.tanh(x)
        return x


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # Flatten
        x = tf.reshape(x, shape=[-1, 7*7*128])
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x


def show_batch(batch):
    img=np.reshape(np.repeat(batch[1, : , : , :], 3, axis=2),
               newshape=(Pic_Len, Pic_Wid, 3))
    plt.imshow(img)
    plt.show()
    return None



# Data preprocess
args = parser.parse_args()
train_filenames, train_labels = list_images(args.train_dir)
#val_filenames, val_labels = list_images(args.val_dir)
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
train_dataset = train_dataset.shuffle(buffer_size=10000)  # do
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





# Build Generator Network
gen_sample = generator(noise_input)


# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(images)
disc_fake = discriminator(gen_sample, reuse=True)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Loss (Labels for real images: 1, for fake images: 0)
# Discriminator Loss for real and fake samples
disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real+tf.random_uniform([batch_size,2], minval=-0.3, maxval=0), labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake+tf.random_uniform([batch_size,2], minval=0, maxval=0.3), labels=tf.zeros([batch_size], dtype=tf.int32)))
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


saver = tf.train.Saver()
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start Training
# Start a new TF session
sess = tf.Session()

# Creat a summary to monitor cost
tf.summary.scalar('g_loss', gen_loss)
tf.summary.scalar('d_loss', disc_loss)
# merger all summaried into a single op
merged_summary_op = tf.summary.merge_all()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    #saver.restore(sess, "/home/cadlab/PycharmProjects/GAN_Architec/model_May25_180000.ckpt")
    # Training
    for i in range(1, num_steps + 1):
        sess.run(train_init_op)
        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, _ = mnist.train.next_batch(batch_size)
        #biatch_x, labels = iterator.get_next()
        #batch_x = np.reshape(batch_x, newshape=[-1, Pic_Len, Pic_Wid, Pic_Cha])
        # Rescale to [-1, 1], the input range of the discriminator
        #batch_x = batch_x * 2. - 1.

        # Discriminator Training
        # Generate noise to feed to the generator
        if i % 2 == 0 or i == 1:
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
            _, dl , summary = sess.run([train_disc, disc_loss, merged_summary_op], feed_dict={noise_input: z, is_training: True})


        # Generator Training
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, gl , summary = sess.run([train_gen, gen_loss, merged_summary_op], feed_dict={noise_input: z, is_training: True})
        summary_writer.add_summary(summary, i)
        if i % 500 == 0 or i == 1:
            #if DEBUG:
            #    show_batch(images)
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        # Generate images from noise, using the generator network.
    #save_path = saver.save(sess, "/home/cadlab/PycharmProjects/GAN_Architec/model_May25_240000.ckpt")
    f, a = plt.subplots(2, 2)  # , figsize=(10, 4))
    for i in range(2):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[2, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z, is_training: False})
        for j in range(2):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 1, axis=2),
                             newshape=(Pic_Len, Pic_Wid, Pic_Cha))
            a[j][i].imshow(img)

    f.show()
