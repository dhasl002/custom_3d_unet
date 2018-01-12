import tensorflow as tf
import numpy as np
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

N = 32
M = 32
P = 7

x = tf.placeholder(tf.float32, shape=[None, N*M*P])
y_ = tf.placeholder(tf.float32, shape=[None, N*M*P, 3])

#STEM
#first convolution
W_conv1 = weight_variable([3, 3, 1, 1, 32])
x_image = tf.reshape(x, [-1, N, M, P, 1])
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
W_conv5_B = weight_variable([1, 7, 1, 64, 64])
h_conv5_B = conv3d_s1(h_conv4_B, W_conv5_B)
#sixth convolution path 2
W_conv6_B = weight_variable([3, 3, 1, 64, 96])
h_conv6_B = conv3d_s1(h_conv5_B, W_conv6_B)
#concatenation
layer1 = tf.concat([h_conv4_A, h_conv6_B],4)
w = tf.Variable(tf.constant(1.,shape=[2,2,4,1,192]))
DeConnv1 = tf.nn.conv3d_transpose(layer1, filter = w, output_shape = [1,N,M,P,1], strides = [1,2,2,2,1], padding = 'SAME')
print(DeConnv1)

r = tf.nn.relu(layer1)

#INCEPTION_A
#first convolution path 1
W_conv1_A = weight_variable([3, 3, 1, 192, 192])
h_conv1_A = conv3d(r, W_conv1_A)
#first convolution path 2- pooling 3x3
h_conv1_B = max_pool_3x3(r)
print(h_conv1_B)
#concat
concat1 = tf.concat([h_conv1_A, h_conv1_B],4)
print(concat1)
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
layer2 = tf.add(h_conv6, concat1)
print(layer2)
w = tf.Variable(tf.constant(1.,shape=[4,4,6,1,384]))
DeConnv2 = tf.nn.conv3d_transpose(layer2, filter = w, output_shape = [1,N,M,P,1], strides = [1,4,4,4,1], padding = 'SAME')


r2 = tf.nn.relu(layer2)

#REDUCTION A
#first pool path 1
h_conv1_A = max_pool_3x3(r2)
print(h_conv1_A)
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
print(r3)

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
#concatenation
concat1 = tf.concat([h_conv3_B, h_conv1_A], 4)
print(concat1)
#second convolution path 1
W_conv2_A = weight_variable([1, 1, 1, 256, 896])
h_conv2_A = conv3d_s1(concat1, W_conv2_A)
#dilation layer1
h_conv4 = conv3d_dilation(h_conv2_A, 800)
#residual addition
layer4 = tf.add(h_conv4, r3)
w = tf.Variable(tf.constant(1.,shape=[8,8,7,1,800]))
DeConnv3 = tf.nn.conv3d_transpose(layer4, filter = w, output_shape = [1,N,M,P,1], strides = [1,8,8,8,1], padding = 'SAME')
print(Deconv3)

r4 = tf.nn.relu(layer4)


#Reduction B
#first convolution path 1
W_conv1_A = weight_variable([1, 1, 1, 800, 192])
h_conv1_A = conv3d_s1(r4, W_conv1_A)
#first maxpool path 2
h_conv1_B = max_pool_3x3(r4)
#first convolution path 3
W_conv1_C = weight_variable([1, 1, 1, 800, 256])
h_conv1_C = conv3d_s1(r4, W_conv1_C)
#second convolution path 1
W_conv2_A = weight_variable([3, 3, 1, 192, 192])
h_conv2_A = conv3d(h_conv1_A, W_conv2_A)
#second convolution path 3
W_conv2_C = weight_variable([1, 7, 1, 256, 256])
h_conv2_C = conv3d_s1(h_conv1_C, W_conv2_C)
#third convolution path 3
W_conv3_C = weight_variable([7, 1, 1, 256, 320])
h_conv3_C = conv3d_s1(h_conv2_C, W_conv3_C)
#fourth convolution path 3
W_conv4_C = weight_variable([3, 3, 1, 320, 320])
h_conv4_C = conv3d(h_conv3_C, W_conv4_C)
#concat
layer5 = tf.concat([h_conv4_C, h_conv1_B, h_conv2_A], 4)

r5 = tf.nn.relu(layer5)
print(r5)

#INCEPTION_C
#first convolution path 1
W_conv1_A = weight_variable([1, 1, 1, 1312, 192])
h_conv1_A = conv3d_s1(r5, W_conv1_A)
#first convolution path 2
W_conv1_B = weight_variable([1, 1, 1, 1312, 192])
h_conv1_B = conv3d_s1(r5, W_conv1_A)
#second convolution path 1
W_conv2_A = weight_variable([1, 1, 1, 192, 2048])
h_conv2_A = conv3d_s1(h_conv1_A, W_conv2_A)
#second convolution path 2
W_conv2_B = weight_variable([1, 3, 1, 192, 224])
h_conv2_B = conv3d_s1(h_conv1_B, W_conv2_B)
#third convolution path 2
W_conv3_B = weight_variable([3, 1, 1, 224, 256])
h_conv3_B = conv3d_s1(h_conv2_B, W_conv3_B)
#concat
concat1 = tf.concat([h_conv3_B,h_conv2_A],4)
print(concat1)
#dilation
h_conv4 = conv3d_dilation(concat1, 1312)
layer6 = tf.add(h_conv4, r5)
w = tf.Variable(tf.constant(1.,shape=[16,16,7,1,1312]))
DeConnv4 = tf.nn.conv3d_transpose(layer6, filter = w, output_shape = [1,N,M,P,1], strides = [1,16,16,16,1], padding = 'SAME')

add1 = tf.add(DeConnv1,DeConnv2)
add2 = tf.add(DeConnv3,DeConnv4)
final = tf.add(add1,add2)
final = tf.reshape(final, [1, N*M*P])
print(DeConnv1)
keep_prob = tf.placeholder(tf.float32)
#h_drop = tf.nn.dropout(final, keep_prob)
W_final = weight_variable([N*M*P,N*M*P,3])
b_final = bias_variable([N*M*P,3])
final_conv = tf.tensordot(final, W_final, axes=[[1], [1]]) + b_final

#train model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(final_conv, 2), tf.argmax(y_, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

a = np.zeros((1,N*M*P),dtype = float)
b = np.zeros((1,N*M*P, 3), dtype = float)
it = 0
num = 1
count = 0
maxNum = 0

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   parent = "C:/Users/dhaslam/Downloads/results/post-processing/New_test_samples/list.txt"
   with open(parent) as inf1:
     next(inf1)
     for line5 in inf1:
       line1, maxNum = line5.strip().split(",")
       path = "C:/Users/dhaslam/Downloads/results/post-processing/New_test_samples/" + line1 + "/" + line1 + "-"
       num = 0
       while num < maxNum:
         it = 0
         with open(path + str(num) + ".txt") as inf:
           next(inf)
           num = num + 1
           for line in inf:
             xCoord, yCoord, zCoord, thresh, label = line.strip().split(",")
             xCoord = int(xCoord)
             yCoord = int(yCoord)
             zCoord = int(zCoord)
             thresh = float(thresh)
             label = int(label)
             a[0][it] = thresh
             b[0][it][label] = 1
             it = it + 1
         train_accuracy = accuracy.eval(feed_dict={x: a, y_: b, keep_prob: 1.0})
         print('step %d, training accuracy %g' % (0,train_accuracy))
         train_step.run(feed_dict={x: a, y_: b, keep_prob: .5})
   path2 = "C:/Users/dhaslam/Downloads/results/post-processing/New_test_samples/4XDA/4XDA-"
   it = 0
   num = 0
   while num < 77:
     with open(path2 + str(num) + ".txt") as inf2:
       next(inf2)
       num = num + 1
       it = 0
       for line3 in inf2:
         xCoord, yCoord, zCoord, thresh, label = line3.strip().split(",")
         xCoord = int(xCoord)
         yCoord = int(yCoord)
         zCoord = int(zCoord)
         thresh = float(thresh)
         label = int(label)
         a[0][it] = thresh
         b[0][it][label] = 1
         it = it + 1
     print(sess.run(accuracy, feed_dict={x: a, y_: b, keep_prob: 1.0}))
     temp = sess.run(tf.argmax(final_conv,2), feed_dict={x: a})
     tempPath = "C:/Users/dhaslam/Desktop/results-"
     f1 = open(tempPath + str(num) + ".txt","w+")
     counter = 0
     while counter < 7168:
       f1.write(str(temp[0][counter]))
       counter = counter + 1
