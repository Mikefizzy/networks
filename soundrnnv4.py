import tensorflow as tf
from networks import IRNN
import numpy as np
from numpy import genfromtxt
import sys
import matplotlib.pyplot as plt
class SoundData:

    def __init__(self, file):
        self.file = file
        amplitudes = genfromtxt(self.file, delimiter=',') #load all of the amplitudes from the csv
        uniques = np.unique(amplitudes).tolist() #extract all unique values from the amplitudes v 
        self.vocab = np.size(uniques) #vocab is the number of unique amps

        self.keyToAmp = {} #dictionary that converts index to Amp
        self.ampToKey = {}
        for i in range(np.size(uniques)):
            self.keyToAmp.update({i: uniques[i]})

        for i in range(np.size(uniques)):
            self.ampToKey.update({uniques[i]:i })

        self.max = self.vocab-1
        #self.data = ampsToIndexes(amplitudes)
        #self.max = np.max(self.data)
    def ampsToIndexes(self, x):
        indexes = [] #list containing all values converted to keys
        for a in x:
            indexes.append(self.ampToKey.get(a))
        return np.array(indexes)

    def generateTrainingData(self, memory, index = 0):
        chunk = genfromtxt(self.file + '-' + str(index), delimiter=',')
        chunk = self.ampsToIndexes(chunk)
        x = []
        y = []

        for i in range(np.size(chunk)-memory):
            x.append(chunk[i:i+memory])
            y.append(chunk[i+memory])

        y = np.array(y)
        x = self.normalize(np.reshape(x, [np.size(x,0), memory, 1]))

        return x,y

    def normalize(self, x):
        a =  (x)/(self.max)
        #a = (a-np.mean(a))/np.std(a)
        return a

    def denormalize(self, x):
        return x*self.max





soundata = SoundData('C:/Users/mikef/Desktop/sounds/supa.csv')
arc = [1,682,soundata.vocab]
memory = 250
nSplits = 300
epochs = 100
startIndex = 0
minibatches = int(1000)

splitRatio = 1 + 1/minibatches
net = IRNN(arc, memory, onehot = soundata.vocab)


costFig = plt.figure()
costPlot = costFig.add_subplot(111)

accuracyFig = plt.figure()
accuracyPlot = accuracyFig.add_subplot(111)


barSize = 80
if(minibatches<barSize):

    barSize = minibatches
changeBarModulo = int(minibatches/barSize)+1

plt.ion()

data_x, data_y = soundata.generateTrainingData(memory, int(nSplits/2))
batch_size = int(np.size(data_x, 0)/ minibatches)
stride = int(np.size(data_y)/batch_size)
sample_x = data_x[::stride]
sample_y = data_y[::stride]

costs = []
accs = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(epochs):
    start = startIndex
    if(i>0):
        start = 0

    for p in range(start, nSplits):
        data_x, data_y = soundata.generateTrainingData(memory, p)

        sys.stdout.write("[%s]" % (" " * barSize))
        sys.stdout.flush()
        sys.stdout.write("\b" * (barSize+1))

        for j in range(minibatches):
            batch_x = data_x[j*batch_size: (j+1)*batch_size]
            batch_y = data_y[j*batch_size: (j+1)*batch_size]

            sess.run(net.optimizer, feed_dict={net.x: batch_x, net.y: batch_y})
            if (j%changeBarModulo == 0):
                sys.stdout.write("|")
                sys.stdout.flush()

        sys.stdout.write("\n")

        if(p%5==0):
            print("should save")

        trainingAccuracy = sess.run(net.accuracy, feed_dict = {net.x: sample_x, net.y: sample_y})
        trainingCost = sess.run(net.cost, feed_dict={net.x: sample_x, net.y: sample_y})


        costs.append(trainingCost)
        accs.append(trainingAccuracy)

        print("epoch #" + str(i)+ " training iter: " +str((i*nSplits + p) * minibatches) + "     cost: " + str(trainingCost) + "    accuracy: " + str(trainingAccuracy))
            
        costPlot.clear()
        costPlot.plot(costs, color = "blue")

        accuracyPlot.clear()
        accuracyPlot.plot(accs, color = "red")



        plt.pause(0.00001)
