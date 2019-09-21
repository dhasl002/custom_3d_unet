import tensorflow as tf
import numpy as np

def conv_3d(tempX, tempW):
  return tf.nn.conv_3d(tempX, tempW, strides=[1, 1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def max_pool_2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1], padding='SAME')

def model(x_image):
  with tf.name_scope("Layer1_analysis"):
    W_conv1 = weight_variable([3, 3, 3, 1, 32])
    h_conv1 = conv_3d(x_image, W_conv1)
    norm_1 = tf.contrib.layers.batch_norm(h_conv1)
    relu_1 = tf.nn.relu(norm_1)

    W_conv2 = weight_variable([3, 3, 3, 32, 64])
    h_conv2 = conv_3d(relu_1, W_conv2)
    norm_2 = tf.contrib.layers.batch_norm(h_conv2)
    relu_2 = tf.nn.relu(norm_2)

  pool1 = max_pool_2x2(relu_2)
  with tf.name_scope("Layer2_analysis"):
    W_conv3 = weight_variable([3, 3, 3, 64, 64])
    h_conv3 = conv_3d(pool1, W_conv3)
    norm_3 = tf.contrib.layers.batch_norm(h_conv3)
    relu_3 = tf.nn.relu(norm_3)

    W_conv4 = weight_variable([3, 3, 3, 64, 128])
    h_conv4 = conv_3d(relu_3, W_conv4)
    norm_4 = tf.contrib.layers.batch_norm(h_conv4)
    relu_4 = tf.nn.relu(norm_4)

  pool2 = max_pool_2x2(relu_4)
  with tf.name_scope("Layer3_analysis"):
    W_conv5 = weight_variable([3, 3, 3, 128, 128])
    h_conv5 = conv_3d(pool2, W_conv5)
    norm_5 = tf.contrib.layers.batch_norm(h_conv5)
    relu_5 = tf.nn.relu(norm_5)

    W_conv6 = weight_variable([3, 3, 3, 128, 256])
    h_conv6 = conv_3d(relu_5, W_conv6)
    norm_6 = tf.contrib.layers.batch_norm(h_conv6)
    relu_6 = tf.nn.relu(norm_6)

  pool3 = max_pool_2x2(relu_6)

  with tf.name_scope("bottom layer"):
    W_conv7 = weight_variable([3, 3, 3, 256, 256])
    h_conv7 = conv_3d(pool3, W_conv7)
    norm_7 = tf.contrib.layers.batch_norm(h_conv7)
    relu_7 = tf.nn.relu(norm_7)

    W_conv8 = weight_variable([3, 3, 3, 256, 512])
    h_conv8 = conv_3d(relu_7, W_conv8)
    norm_8 = tf.contrib.layers.batch_norm(h_conv8)
    relu_8 = tf.nn.relu(norm_8)

  up_sample1 = tf.layers.conv_3d_transpose(relu_8, filters = 512, kernel_size = (2,2,2), strides = (2,2,2))

  concat1 = tf.concat([relu_6, up_sample1],4)

  with tf.name_scope("Layer3_synthesis"):
    W_conv9 = weight_variable([3, 3, 3, 768, 256])
    h_conv9 = conv_3d(concat1, W_conv9)
    norm_9 = tf.contrib.layers.batch_norm(h_conv9)
    relu_9 = tf.nn.relu(norm_9)

    W_conv10 = weight_variable([3, 3, 3, 256, 256])
    h_conv10 = conv_3d(relu_9, W_conv10)
    norm_10 = tf.contrib.layers.batch_norm(h_conv10)
    relu_10 = tf.nn.relu(norm_10)

  up_sample2 = tf.layers.conv_3d_transpose(relu_10, filters = 256, kernel_size = (2,2,2), strides = (2,2,2))

  concat2 = tf.concat([relu_4, up_sample2],4)

  with tf.name_scope("Layer2_synthesis"):
    W_conv11 = weight_variable([3, 3, 3, 384, 128])
    h_conv11 = conv_3d(concat2, W_conv11)
    norm_11 = tf.contrib.layers.batch_norm(h_conv11)
    relu_11 = tf.nn.relu(norm_11)

    W_conv12 = weight_variable([3, 3, 3, 128, 128])
    h_conv12 = conv_3d(relu_11, W_conv12)
    norm_12 = tf.contrib.layers.batch_norm(h_conv12)
    relu_12 = tf.nn.relu(norm_12)

  up_sample3 = tf.layers.conv_3d_transpose(relu_12, filters = 128, kernel_size = (2,2,2), strides = (2,2,2))

  concat3 = tf.concat([relu_2, up_sample3],4)

  with tf.name_scope("Layer1_synthesis"):
    W_conv13 = weight_variable([3, 3, 3, 192, 64])
    h_conv13 = conv_3d(concat3, W_conv13)
    norm_13 = tf.contrib.layers.batch_norm(h_conv13)
    relu_13 = tf.nn.relu(norm_13)

    W_conv14 = weight_variable([3, 3, 3, 64, 64])
    h_conv14 = conv_3d(relu_13, W_conv14)
    norm_14 = tf.contrib.layers.batch_norm(h_conv14)
    relu_14 = tf.nn.relu(norm_14)

    last_wt = weight_variable([1, 1, 1, 64, 3])
    last_conv = conv_3d(relu_14, last_wt)

  final_conv = tf.reshape(last_conv, [-1, 3])
  return final_conv

