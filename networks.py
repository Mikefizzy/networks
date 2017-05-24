#irnn
import tensorflow as tf
import numpy as np

class IRNN:
    def __init__(self, arc, memory, optimizerMethod = tf.train.AdamOptimizer(10**-3), alpha = 0.5, activation ='softmax', cost = 'cross_entropy', onehot = 0):
        #arc is the architecture array ex. [number of inputs, width, width, width, number of outputs]
        #alpha is a contant multiplied by the identity matrix, low for short memory, 1 for long term
        self.arc = arc;
        self.memory = memory;

        nInputs = arc[0]
        nOutputs = arc[-1]
        width = arc[1]
        rnn_layers = len(arc)-2

        self.x = tf.placeholder(tf.float32, [None, memory, nInputs])
        if(onehot>0):
            self.y = tf.placeholder(tf.int32, [None])
            y = tf.one_hot(self.y, onehot)
        else:
            self.y = tf.placeholder(tf.float32, [None, nOutputs])
            y = self.y;
        batch_size = tf.shape(self.x)[0]

        w1 = [tf.Variable(tf.truncated_normal([nInputs, width], 0.0, 0.01, dtype=tf.float32))] #list of weight matrices for each lstm layer
        b1 = [tf.Variable(np.zeros(width), dtype = tf.float32)]

        for i in range(len(arc)-3): #adds the remaining parameters to the lists
            w1.append(tf.Variable(tf.truncated_normal([width, width], 0.0, 0.01, dtype=tf.float32))) 
            b1.append(tf.Variable(np.zeros(width), dtype = tf.float32))
        rnn_w = [tf.Variable(np.identity(width)*alpha, dtype = tf.float32)]*rnn_layers #list of state weight matrices which propogate through time
        '''
        hidden = tf.reshape(layer_state, [-1,arc[i]]) #[a,b,c]->[a*b, c]
        hidden = tf.add(tf.matmul(hidden,w[i]), b[i]) 
        hidden = tf.reshape(hidden, [self.memory, batch_size, width])
        '''
        def condition(j,n,w,li,s,a):#index, memory, state weight matrix, rnn state, temp array
            return tf.less(j,n)

        x = tf.transpose(self.x, [1,0,2])# [None, memory, nInputs] -> [memory, None, nInputs]
        layer = x;

        for i in range(rnn_layers):

            temp = tf.TensorArray(dtype=tf.float32, size=self.memory) #array which holds state for each time step
            state = tf.zeros([batch_size, width], dtype=tf.float32)# initial state

            def step(j,n,w,li,s,a):
                s = tf.nn.relu(tf.add(tf.matmul(s, w), tf.add(tf.matmul(li[j],w1[i]), b1[i]) ))
                a = a.write(j,s)
                return [j+1, n,w,li,s,a]

            index = 0
            _,_,_,_,_,out = tf.while_loop(condition, step, [index, memory, rnn_w[i], layer,state, temp])

            out = out.stack()
            layer = out



        w2 = tf.Variable(np.random.rand(width, nOutputs), dtype=tf.float32)
        b2 = tf.Variable(np.zeros(nOutputs), dtype = tf.float32)
        out = layer[-1]
        out = tf.add(tf.matmul(out, w2), b2)
        if(activation=='softmax'):
            self.out = tf.nn.softmax(out)
        elif(activation=='sigmoid'):
            self.out = tf.sigmoid(out)
        else:
            self.out = out
        if(cost=='cross_entropy'):
            self.cost = tf.reduce_mean(-y*tf.log(self.out) - (1-y)*tf.log(1-self.out)) #cross entropy
        elif(cost =='mean_square'):
            self.cost = tf.reduce_mean(tf.pow(self.out-y,2))

        correct_pred = tf.equal(tf.argmax(self.out,1), tf.argmax(y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.optimizer = optimizerMethod.minimize(self.cost)

class LSTM:
    def __init__(self, arc, memory,dropout = 0, optimize_method = tf.train.AdamOptimizer(0.001), activation ='softmax', cost = 'cross_entropy', onehot = 0):
        self.arc = arc
        self.memory = memory
        self.lstmLayers = self.arc[1:-1] #the amount of units per stacked lstm layer

        self.x = tf.placeholder(tf.float32, [None, self.memory, arc[0]])
        self.y = tf.placeholder(tf.float32, [None, arc[-1]])
        if(onehot>0):
            self.y = tf.placeholder(tf.int32, [None])
            y = tf.one_hot(self.y, arc[-1])
        else:
            y = self.y
        x = tf.unstack(self.x, axis = 1)

        cells = []
        for i in self.lstmLayers:
            cell = tf.contrib.rnn.BasicLSTMCell(i, forget_bias = 1.0)
            if(dropout>0):#DropoutWrapper
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1-dropout)
            cells.append(cell)

        self.cell = tf.contrib.rnn.MultiRNNCell(cells)
        self.outputs, self.states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)

        self.W2 = tf.Variable(tf.random_normal([self.arc[-2], self.arc[-1]]))
        self.B2 = tf.Variable(tf.random_normal([self.arc[-1]]))
        #self.W1 = tf.Variable(tf.random_normal([self.arc[-3], self.arc[-2]]))
        #self.B1 = tf.Variable(tf.random_normal([self.arc[-2]]))

        #out = tf.matmul((tf.nn.relu (tf.matmul(self.outputs[-1], self.W1) + self.B1)), self.W2) + self.B2
        out  = tf.matmul(self.outputs[-1], self.W2) + self.B2
        if(activation=='softmax'):
            self.out = tf.nn.softmax(out)
        elif(activation=='sigmoid'):
            self.out = tf.sigmoid(out)
        else:
            self.out = out

        if(onehot>0):
            y = tf.cast(y, tf.float32)
        if(cost=='cross_entropy'):
            self.cost = tf.reduce_mean(-y*tf.log(self.out) - (1-y)*tf.log(1-self.out)) #cross entropy
        elif(cost =='mean_square'):
            self.cost = tf.reduce_mean(tf.pow(self.out-y,2))

        correct_pred = tf.equal(tf.argmax(self.out,1), tf.argmax(y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #self.accuracy = tf.reduce_mean(tf.sqrt(tf.cast(tf.square(tf.argmax(self.out, 1) - tf.argmax(y,1)), tf.float32) ))
        self.out_argmax = tf.argmax(self.out, 1)

        self.grads = optimize_method.compute_gradients(self.cost)
        capped = [(tf.clip_by_value(grad, -0.3, 0.3), var) for grad, var in self.grads]
        self.optimizer = optimize_method.apply_gradients(capped)
        #self.optimizer = optimize_method.minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.out,1), tf.argmax(self.y,1))