import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
sess = tf.InteractiveSession()

def conv3d_dilation(tempX, tempFilter):
  return tf.layers.conv3d(tempX, filters = tempFilter, kernel_size=[3, 3, 1], strides=1, padding='SAME', dilation_rate = 2)

def conv3d(tempX, tempW):
  return tf.nn.conv3d(tempX, tempW, strides=[1, 2, 2, 2, 1], padding='SAME')

def conv3d_s1(tempX, tempW):
  return tf.nn.conv3d(tempX, tempW, strides=[1, 1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def max_pool_3x3(x):
  return tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1],strides=[1, 2, 2, 2, 1], padding='SAME')

def model(x_image, y_, N, M, P):
  #STEM
  #first convolution
  W_conv1 = weight_variable([3, 3, 1, 1, 32])
  h_conv1 = conv3d(x_image, W_conv1)
  #second convolution
  W_conv2 = weight_variable([3, 3, 4, 32, 64])
  h_conv2 = conv3d_s1(h_conv1, W_conv2)
  #third convolution path 1
  W_conv3_A = weight_variable([1, 1, 1, 64, 64])
  h_conv3_A = conv3d_s1(h_conv2, W_conv3_A)
  #third convolution path 2
  W_conv3_B = weight_variable([1, 1, 1, 64, 64])
  h_conv3_B = conv3d_s1(h_conv2, W_conv3_B)
  #fourth convolution path 1
  W_conv4_A = weight_variable([3, 3, 1, 64, 96])
  h_conv4_A = conv3d_s1(h_conv3_A, W_conv4_A)
  #fourth convolution path 2
  W_conv4_B = weight_variable([1, 7, 1, 64, 64])
  h_conv4_B = conv3d_s1(h_conv3_B, W_conv4_B)
  #fifth convolution path 2
  W_conv5_B = weight_variable([7, 1, 1, 64, 64])
  h_conv5_B = conv3d_s1(h_conv4_B, W_conv5_B)
  #sixth convolution path 2
  W_conv6_B = weight_variable([3, 3, 1, 64, 96])
  h_conv6_B = conv3d_s1(h_conv5_B, W_conv6_B)
  #concatenation
  layer1 = tf.concat([h_conv4_A, h_conv6_B],4)
  w = tf.Variable(tf.constant(1.,shape=[2,2,4,1,192]))
  DeConnv1 = tf.nn.conv3d_transpose(layer1, filter = w, output_shape = tf.shape(x_image), strides = [1,2,2,2,1], padding = 'SAME')

  r = tf.nn.relu(layer1)

  #INCEPTION_A
  #first convolution path 1
  W_conv1_A = weight_variable([3, 3, 1, 192, 192])
  h_conv1_A = conv3d(r, W_conv1_A)
  #first convolution path 2- pooling 3x3
  h_conv1_B = max_pool_3x3(r)
  #concat
  concat1 = tf.concat([h_conv1_A, h_conv1_B],4)
  #second convolution path 1
  W_conv2_A = weight_variable([1, 1, 1, 384, 32])
  h_conv2_A = conv3d_s1(concat1, W_conv2_A)
  #second convolution path 2
  W_conv2_B = weight_variable([1, 1, 1, 384, 32])
  h_conv2_B = conv3d_s1(concat1, W_conv2_B)
  #second convolution path 3
  W_conv2_C = weight_variable([1, 1, 1, 384, 32])
  h_conv2_C = conv3d_s1(concat1, W_conv2_C)
  #third convolution path 2
  W_conv3_B = weight_variable([3, 3, 1, 32, 32])
  h_conv3_B = conv3d_s1(h_conv2_B, W_conv3_B)
  #third convolution path 3
  W_conv3_C = weight_variable([3, 3, 1, 32, 48])
  h_conv3_C = conv3d_s1(h_conv2_C, W_conv3_C)
  #fourth convolution path 3
  W_conv4_C = weight_variable([3, 3, 1, 48, 64])
  h_conv4_C = conv3d_s1(h_conv3_C, W_conv4_C)
  #concat
  concat2 = tf.concat([h_conv4_C, h_conv3_B, h_conv2_A],4)
  #convolution after paths merge
  W_conv5 = weight_variable([3, 3, 1, 128, 384])
  h_conv5 = conv3d_s1(concat2, W_conv5)
  #last convolution in inceptionA, this is a dialated convolution
  h_conv6 = conv3d_dilation(h_conv5, 384)
  #residual learning added to last convolution
  #layer2 = h_conv6
  layer2 = tf.add(h_conv6, concat1)
  w = tf.Variable(tf.constant(1.,shape=[4,4,6,1,384]))
  DeConnv2 = tf.nn.conv3d_transpose(layer2, filter = w, output_shape = tf.shape(x_image), strides = [1,4,4,4,1], padding = 'SAME')


  r2 = tf.nn.relu(layer2)

  #REDUCTION A
  #first pool path 1
  h_conv1_A = max_pool_3x3(r2)
  #first convolution path 2
  W_conv1_B = weight_variable([3, 3, 1, 384, 32])
  h_conv1_B = conv3d(r2, W_conv1_B)
  #first convolution path 3
  W_conv1_C = weight_variable([1, 1, 1, 384, 256])
  h_conv1_C = conv3d_s1 (r2, W_conv1_C)
  #second convolution path 3
  W_conv2_C = weight_variable([3, 3, 1, 256, 256])
  h_conv2_C = conv3d_s1(h_conv1_C, W_conv2_C)
  #third convolution path 3
  W_conv3_C = weight_variable([3, 3, 1, 256, 384])
  h_conv3_C = conv3d(h_conv2_C, W_conv3_C)
  #last step of reduction a, concat
  layer3 = tf.concat([h_conv1_B, h_conv1_A, h_conv3_C],4)

  r3 = tf.nn.relu(layer3)

  #INCEPTION_B
  #first convolution path 1
  W_conv1_A = weight_variable([1, 1, 1, 800, 128])
  h_conv1_A = conv3d_s1(r3, W_conv1_A)
  #first convolution path 2
  W_conv1_B = weight_variable([1, 1, 1, 800, 128])
  h_conv1_B = conv3d_s1(r3, W_conv1_B)
  #second convolution path 2
  W_conv2_B = weight_variable([1, 7, 1, 128, 128])
  h_conv2_B = conv3d_s1(h_conv1_B, W_conv2_B)
  #third convolution path 2
  W_conv3_B = weight_variable([7, 1, 1, 128, 128])
  h_conv3_B = conv3d_s1(h_conv2_B, W_conv3_B)
  #second convolution path 1
  W_conv2_A = weight_variable([1, 1, 1, 128, 896])
  h_conv2_A = conv3d_s1(h_conv1_A, W_conv2_A)
  #concatenation
  concat1 = tf.concat([h_conv3_B, h_conv2_A], 4)
  #dilation layer1
  h_conv4 = conv3d_dilation(concat1, 800)
  #residual addition
  #layer4 = h_conv4
  layer4 = tf.add(h_conv4, r3)
  w = tf.Variable(tf.constant(1.,shape=[8,8,7,1,800]))
  DeConnv3 = tf.nn.conv3d_transpose(layer4, filter = w, output_shape = tf.shape(x_image), strides = [1,8,8,8,1], padding = 'SAME')

  r4 = tf.nn.relu(layer4)


  #Reduction B
  #first convolution path 1
  #W_conv1_A = weight_variable([1, 1, 1, 800, 192])
  #h_conv1_A = conv3d_s1(r4, W_conv1_A)
  #first maxpool path 2
  h_conv1_B = max_pool_3x3(r4)
  #first convolution path 3
  W_conv1_C = weight_variable([1, 1, 1, 800, 256])
  h_conv1_C = conv3d_s1(r4, W_conv1_C)
  #second convolution path 1
  W_conv2_A = weight_variable([3, 3, 1, 800, 32])
  h_conv2_A = conv3d(r4, W_conv2_A)
  #second convolution path 3
  W_conv2_C = weight_variable([3, 3, 1, 256, 256])
  h_conv2_C = conv3d_s1(h_conv1_C, W_conv2_C)
  #third convolution path 3
  #W_conv3_C = weight_variable([7, 1, 1, 256, 320])
  #h_conv3_C = conv3d_s1(h_conv2_C, W_conv3_C)
  #fourth convolution path 3
  W_conv4_C = weight_variable([3, 3, 1, 256, 384])
  h_conv4_C = conv3d(h_conv2_C, W_conv4_C)
  #concat
  layer5 = tf.concat([h_conv4_C, h_conv1_B, h_conv2_A], 4)

  r5 = tf.nn.relu(layer5)

  #INCEPTION_C
  #first convolution path 1
  W_conv1_A = weight_variable([1, 1, 1, 1216, 128])
  h_conv1_A = conv3d_s1(r5, W_conv1_A)
  #first convolution path 2
  W_conv1_B = weight_variable([1, 1, 1, 1216, 128])
  h_conv1_B = conv3d_s1(r5, W_conv1_B)
  #second convolution path 1
  W_conv2_A = weight_variable([1, 1, 1, 128, 896])
  h_conv2_A = conv3d_s1(h_conv1_A, W_conv2_A)
  #second convolution path 2
  W_conv2_B = weight_variable([1, 7, 1, 128, 128])
  h_conv2_B = conv3d_s1(h_conv1_B, W_conv2_B)
  #third convolution path 2
  W_conv3_B = weight_variable([7, 1, 1, 128, 128])
  h_conv3_B = conv3d_s1(h_conv2_B, W_conv3_B)
  #concat
  concat1 = tf.concat([h_conv3_B,h_conv2_A],4)
  #dilation
  h_conv4 = conv3d_dilation(concat1, 1216)
  #layer6 = h_conv4
  layer6 = tf.add(h_conv4, r5)
  w = tf.Variable(tf.constant(1.,shape=[16,16,7,1,1216]))
  DeConnv4 = tf.nn.conv3d_transpose(layer6, filter = w, output_shape = tf.shape(x_image), strides = [1,16,16,16,1], padding = 'SAME')

  add1 = tf.add(DeConnv1,DeConnv2)
  add2 = tf.add(DeConnv3,DeConnv4)
  final = tf.add(add1,add2)

  final= tf.reshape(final, [-1, N*M*P])

  W_final = weight_variable([N*M*P,N*M*P,3])
  b_final = bias_variable([N*M*P,3])

  final_conv = tf.tensordot(final, W_final, axes=[[1], [1]]) + b_final
  return final_conv
