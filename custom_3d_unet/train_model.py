import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from custom_3d_unet import model
from random import *
from os import listdir
from os.path import isfile, join
from utils import *

def train(batch_size, patch_width, patch_height, patch_depth):
  data_files = [join("../data/", f) for f in listdir("../data/") if isfile(join("../data/", f))]
  for i in range(0, len(data_files)):
    a, b = read_mrc_image_data(data_files[i], patch_width, patch_height, patch_depth, batch_size, i)
    if(i % batch_Size == batch_Size - 1):
      train_batch(a, b)

def train_batch(a, b):
  keep_prob = tf.placeholder(tf.float32)
  current_loss = cross_entropy.eval(feed_dict={x_image: a, y_: b})
  train_accuracy = accuracy.eval(feed_dict={x_image: a, y_: b, keep_prob: .5})
  train_step.run(feed_dict={x_image: a, y_: b, keep_prob: .5})
  print_batch_stats(train_accuracy, current_loss)

if __name__ == "__main__":
  patch_width = 48
  patch_height = 48
  patch_depth = 48
  num_labels = 3
  batch_Size = 4
  epochs = 100000

  x_image = tf.placeholder(tf.float32, shape=[None, patch_width, patch_height, patch_depth, 1])
  y_ = tf.placeholder(tf.float32, shape=[None, None, num_labels])
  modelResult = model(x_image)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelResult, labels=y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(modelResult, 1), tf.argmax(tf.reshape(y_, [-1, num_labels]), 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for curEpoch in range(1, epochs):
      print('epoch Number: ' + str(curEpoch))
      train(batch_Size, patch_width, patch_height, patch_depth)

