import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tf_model import model
sess = tf.InteractiveSession()

def printEpoch(curEpoch):
  print('epoch Number: ' + str(curEpoch))
  fStat.write('epoch Number: ' + str(curEpoch) + '\n')

def printBatchStats(train_accuracy, CurCross):
  fStat.write('training accuracy %g' % (train_accuracy))
  fStat.write('\n')
  fStat.write('Loss %g' % (CurCross))
  fStat.write('\n')
  print('training accuracy %g' % (train_accuracy))
  print('Loss %g' % (CurCross))

def train(trainingListLocation, batch_Size):
  #open list that contains information about training (protein name and number of files)
  with open(trainingListLocation) as trainingList:
    next(trainingList)
	#extract protein name and number of files
    for newLine in trainingList:
      proteinName, numFiles = newLine.strip().split(",")
      path = "/home/dhaslam/New_test_samples2/RotatedSet/" + proteinName + "/" + proteinName + "-"
      trainProtein(path, numFiles, batch_Size)

def trainProtein(path, numFiles, batch_Size):
  for currFileNum in range(0, int(numFiles)):
    it = 0
    if (int(currFileNum)%batch_Size == 0):
      a = np.zeros((batch_Size,N,M,P,1),dtype = float)
      b = np.zeros((batch_Size,N*M*P,3), dtype = float)
    with open(path + str(currFileNum) + ".txt") as inf:
      for line in inf:
        xCoord, yCoord, zCoord, thresh, label = line.strip().split(",")
        xCoord = int(xCoord)
        yCoord = int(yCoord)
        zCoord = int(zCoord)
        thresh = float(thresh)
        label = int(label)	
        a[int(currFileNum)%batch_Size][it%N][it%M][it%P][0] = thresh					   
        b[int(currFileNum)%batch_Size][it][label] = 1
        it = it + 1
    if(int(currFileNum)%batch_Size == (batch_Size)-1):
      runTrainingBatch(a, b)

def runTrainingBatch(a, b):
  keep_prob = tf.placeholder(tf.float32)
  CurCross = cross_entropy.eval(feed_dict={x_image: a, y_: b})
  train_accuracy = accuracy.eval(feed_dict={x_image: a, y_: b, keep_prob: .5})
  train_step.run(feed_dict={x_image: a, y_: b, keep_prob: .5})
  printBatchStats(train_accuracy, CurCross)

def test(batch_Size, final_conv):
  it = 0
  num = 0
  keep_prob = tf.placeholder(tf.float32)
  #hard coded for amount of test data
  while int(num) < 40:
    numLines = 0
    #check how many lines are in test file
    xMin = 100000000
    xMax = -1
    yMin = 100000000
    yMax = -1
    zMin = 100000000
    zMax = -1
    with open(path2 + str(num) + ".txt") as inf3:
      for line4 in inf3:
        numLines = numLines + 1
        xCoord, yCoord, zCoord, thresh, label = line4.strip().split(",")
        xCoord = int(xCoord)
        yCoord = int(yCoord)
        zCoord = int(zCoord)
        if xCoord < xMin:
          xMin = xCoord
        if yCoord < yMin:
          yMin = yCoord
        if zCoord < zMin:
          zMin = zCoord
        if xCoord > xMax:
          xMax = xCoord
        if yCoord > yMax:
          yMax = yCoord
        if zCoord > zMax:
          zMax = zCoord
    with open(path2 + str(num) + ".txt") as inf2:
      a = np.zeros((batch_Size,xMax-xMin+1, yMax-yMin+1, zMax-zMin+1, 1),dtype = float)
      b = np.zeros((batch_Size,numLines, 3), dtype = float)
      axisZ = np.zeros((1,numLines),dtype = float)
      axisY = np.zeros((1,numLines), dtype = float)
      axisX = np.zeros((1,numLines),dtype = float)
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
        axisX[0][it] = xCoord
        axisY[0][it] = yCoord
        axisZ[0][it] = zCoord
        a[int(num)%batch_Size][it%(xMax-xMin+1)][it%(yMax-yMin+1)][it%(zMax-zMin+1)][0]  = thresh
        b[0][it][label] = 1
        it = it + 1
    print(sess.run(accuracy, feed_dict={x_image: a, y_: b, keep_prob: 1.0}))
    temp = sess.run(tf.argmax(final_conv,2), feed_dict={x_image: a})
    writePredictionsToFile(temp, numLines, num, axisX, axisY, axisZ)

def writePredictionsToFile(temp, numLines, num, axisX, axisY, axisZ):
  f1 = open(tempPath + str(num) + ".txt","w+")
  for i in range(0, numLines):
    f1.write(str(int(axisX[0][i])) + " " + str(int(axisY[0][i])) + " " + str(int(axisZ[0][i])) + " " + str(temp[0][i]) + "\r\n")


N = 32 #x dimension of training patch size
M = 32 #y dimension of training patch size
P = 7 #z dimension of training patch size
batch_Size = 30
epochs = 300
outputStats = "/home/dhaslam/New_test_samples2/RotatedSet/labels/Statistics.txt"
trainingListLocation = "/home/dhaslam/New_test_samples2/RotatedSet/list.txt"
path2 = "/home/dhaslam/New_test_samples2/RotatedSet/4XDA/4XDA-"
tempPath = "/home/dhaslam/New_test_samples2/RotatedSet/labels/results-"
x_image = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
y_ = tf.placeholder(tf.float32, shape=[None, None, 3])
  
modelResult = model(x_image, y_, N, M, P) #sets modelResults to the final result of the model 

#train model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelResult, labels=y_))
train_step = tf.train.RMSPropOptimizer(1e-4, epsilon = .1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(modelResult, 2), tf.argmax(y_, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   fStat = open(outputStats,"w+")
   for curEpoch in range(1, epochs):
     printEpoch(curEpoch) 
     train(trainingListLocation, batch_Size)
     test(batch_Size, modelResult)
   fStat.close()
