import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tf_Unet import model
from tensorflow.python import debug as tf_debug
from random import *
sess = tf.InteractiveSession()

def printEpoch(curEpoch):
  print('epoch Number: ' + str(curEpoch))

def printBatchStats(train_accuracy, CurCross):
  print('training accuracy %g' % (train_accuracy))
  print('Loss %g' % (CurCross))

def train(trainingListLocation, batch_Size):
  proteinList = []
  numTraining = []
  #open list that contains information about training (protein name and number of files)
  with open(trainingListLocation) as trainingList:
    next(trainingList)
	#extract protein name and number of files
    for newLine in trainingList:
      proteinName, numFiles = newLine.strip().split(",")
      proteinList.append(proteinName)
      numTraining.append(numFiles)
  trainProtein(numFiles, batch_Size, proteinList, numTraining)

def trainProtein(numFiles, batch_Size, proteinList, numTraining):
  for currIt in range(0, (len(proteinList)*int(numTraining[0]))/6):
    randNum = randint(0, (len(proteinList)-1))
    randNum2 = randint(0, int(numTraining[randNum]))
    path = "/home/dhaslam/New_test_samples2/RotatedSet/" + proteinList[randNum] + "/" + proteinList[randNum] + "_"
    #print(path)
    xMax, xMin, yMax, yMin, zMax, zMin = getMRCDimensions(path + str(randNum2) + "_label.txt")
    xLength = xMax-xMin+1
    yLength = yMax-yMin+1
    zLength = zMax-zMin+1
    numLines = (xLength)*(yLength)*(zLength)
    with open(path + str(randNum2) + "_label.txt") as inf:
      a = np.zeros((1, xLength, yLength, zLength, 1),dtype = float)
      b = np.zeros((1, xLength, yLength, zLength, 3), dtype = float)
      next(inf)
      for line in inf:
        xCoord, yCoord, zCoord, thresh, label = line.strip().split(",")
        xCoord = int(xCoord)
        yCoord = int(yCoord)
        zCoord = int(zCoord) 
        thresh = float(thresh)
        label = int(label)
        #curPos = (xCoord-xMin)*yLength*zLength+(yCoord-yMin)*zLength+(zCoord-zMin)
        a[0][xCoord-xMin][yCoord-yMin][zCoord-zMin][0]  = thresh
        b[0][xCoord-xMin][yCoord-yMin][zCoord-zMin][label] = 1
    if(int(currIt)%batch_Size == (batch_Size)-1):
      randNumX1 = randint(0, (xLength-N))
      randNumY1 = randint(0, (yLength-M))
      randNumZ1 = randint(0, (zLength-P))	  
      a = np.delete(a, slice(0, randNumX1) ,axis=1)
      a = np.delete(a, slice((xLength-randNumX1)-((xLength-N)-randNumX1), (xLength-randNumX1)) ,axis=1)
      a = np.delete(a, slice(0, randNumY1) ,axis=2)
      a = np.delete(a, slice((yLength-randNumY1)-((yLength-M)-randNumY1), (yLength-randNumY1)) ,axis=2)
      a = np.delete(a, slice(0, randNumZ1) ,axis=3)
      a = np.delete(a, slice((zLength-randNumZ1)-((zLength-P)-randNumZ1), (zLength-randNumZ1)) ,axis=3)
	  
      b = np.delete(b, slice(0, randNumX1) ,axis=1)
      b = np.delete(b, slice((xLength-randNumX1)-((xLength-N)-randNumX1), (xLength-randNumX1)) ,axis=1)
      b = np.delete(b, slice(0, randNumY1) ,axis=2)
      b = np.delete(b, slice((yLength-randNumY1)-((yLength-M)-randNumY1), (yLength-randNumY1)) ,axis=2)
      b = np.delete(b, slice(0, randNumZ1) ,axis=3)
      b = np.delete(b, slice((zLength-randNumZ1)-((zLength-P)-randNumZ1), (zLength-randNumZ1)) ,axis=3)	  
      b = np.reshape(b, (-1,N*M*P,3))
      runTrainingBatch(a, b)

def test(batch_Size, best):
  keep_prob = tf.placeholder(tf.float32)
  for currIt in range(0, 1):
    randNum2 = randint(0, 35)
    path = "/home/dhaslam/New_test_samples2/RotatedSet/4XDA/4XDA_"
    xMax, xMin, yMax, yMin, zMax, zMin = getMRCDimensions(path + str(randNum2) + "_label.txt")
    xLength = xMax-xMin+1
    yLength = yMax-yMin+1
    zLength = zMax-zMin+1
    numLines = (xLength)*(yLength)*(zLength)
    with open(path + str(randNum2) + "_label.txt") as inf:
      a = np.zeros((1, xLength, yLength, zLength, 1),dtype = float)
      b = np.zeros((1, xLength, yLength, zLength, 3), dtype = float)
      next(inf)
      for line in inf:
        xCoord, yCoord, zCoord, thresh, label = line.strip().split(",")
        xCoord = int(xCoord)
        yCoord = int(yCoord)
        zCoord = int(zCoord) 
        thresh = float(thresh)
        label = int(label)
        #curPos = (xCoord-xMin)*yLength*zLength+(yCoord-yMin)*zLength+(zCoord-zMin)
        a[0][xCoord-xMin][yCoord-yMin][zCoord-zMin][0]  = thresh
        b[0][xCoord-xMin][yCoord-yMin][zCoord-zMin][label] = 1
    if(int(currIt)%batch_Size == (batch_Size)-1):
      randNumX1 = randint(0, (xLength-N))
      randNumY1 = randint(0, (yLength-M))
      randNumZ1 = randint(0, (zLength-P))	  
      a = np.delete(a, slice(0, randNumX1) ,axis=1)
      a = np.delete(a, slice((xLength-randNumX1)-((xLength-N)-randNumX1), (xLength-randNumX1)) ,axis=1)
      a = np.delete(a, slice(0, randNumY1) ,axis=2)
      a = np.delete(a, slice((yLength-randNumY1)-((yLength-M)-randNumY1), (yLength-randNumY1)) ,axis=2)
      a = np.delete(a, slice(0, randNumZ1) ,axis=3)
      a = np.delete(a, slice((zLength-randNumZ1)-((zLength-P)-randNumZ1), (zLength-randNumZ1)) ,axis=3)
	  
      b = np.delete(b, slice(0, randNumX1) ,axis=1)
      b = np.delete(b, slice((xLength-randNumX1)-((xLength-N)-randNumX1), (xLength-randNumX1)) ,axis=1)
      b = np.delete(b, slice(0, randNumY1) ,axis=2)
      b = np.delete(b, slice((yLength-randNumY1)-((yLength-M)-randNumY1), (yLength-randNumY1)) ,axis=2)
      b = np.delete(b, slice(0, randNumZ1) ,axis=3)
      b = np.delete(b, slice((zLength-randNumZ1)-((zLength-P)-randNumZ1), (zLength-randNumZ1)) ,axis=3)	  
      b = np.reshape(b, (-1,N*M*P,3))
      print(sess.run(accuracy, feed_dict={x_image: a, y_: b, keep_prob: 1.0}))
      predictedLabels = sess.run(tf.argmax(modelResult,1), feed_dict={x_image: a})
      axisZ = np.zeros((1,N*M*P),dtype = float)
      axisY = np.zeros((1,N*M*P),dtype = float)
      axisX = np.zeros((1,N*M*P),dtype = float)
      for x in range(0, N):
        for y in range(0, M):
          for z in range(0, P):
            curPos = (x)*M*P+(y)*P+(z)
            axisX[0][curPos] = x+xMin
            axisY[0][curPos] = y+yMin
            axisZ[0][curPos] = z+zMin
    if tf.less(tf.cast(best, tf.float32), accuracy) is not None:
      writePredictionsToFile(predictedLabels, N*M*P, currIt, axisX, axisY, axisZ)
      f5 = open(accPath,"w+")
      f5.write(str(accuracy))
      f5.close()
      best = accuracy
  return best

def writePredictionsToFile(predictedLabels, numLines, index, axisX, axisY, axisZ):
  path = tempPath + str(index) + ".txt"
  f1 = open(path,"w+")
  for i in range(0, numLines):
    f1.write(str(int(axisX[0][i])) + " " + str(int(axisY[0][i])) + " " + str(int(axisZ[0][i])) + " " + str(predictedLabels[i]) + "\r\n")

def runTrainingBatch(a, b):
  keep_prob = tf.placeholder(tf.float32)
  CurCross = cross_entropy.eval(feed_dict={x_image: a, y_: b})
  train_accuracy = accuracy.eval(feed_dict={x_image: a, y_: b, keep_prob: .5})
  train_step.run(feed_dict={x_image: a, y_: b, keep_prob: .5})
  printBatchStats(train_accuracy, CurCross)

def getMRCDimensions(fileToOpen):
  xMin = 100000000
  xMax = -1
  yMin = 100000000
  yMax = -1
  zMin = 100000000
  zMax = -1
  with open(fileToOpen) as inf3:
    for line4 in inf3:
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
  return xMax, xMin, yMax, yMin, zMax, zMin


N = 48
M = 48
P = 48
batch_Size = 1
epochs = 10000000
trainingListLocation = "/home/dhaslam/New_test_samples2/RotatedSet/listUnet.txt"
tempPath = "/home/dhaslam/New_test_samples2/RotatedSet/labels/results-UNET-"
accPath = "/home/dhaslam/New_test_samples2/RotatedSet/labels/results-UNET-acc"
x_image = tf.placeholder(tf.float32, shape=[1, 48, 48, 48, 1])
y_ = tf.placeholder(tf.float32, shape=[None, None, 3])
best = 0

modelResult = model(x_image) #sets modelResults to the final result of the model 

#train model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelResult, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(modelResult, 1), tf.argmax(tf.reshape(y_, [-1, 3]), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
   for curEpoch in range(1, epochs):
     printEpoch(curEpoch) 
     train(trainingListLocation, batch_Size)
     best = test(batch_Size, best)
   fStat.close()
