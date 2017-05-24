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
		x = tf.transpose(self.x, [1,0,2])# [None, memory, nInputs] -> [memory, None, nInputs]

		w = [tf.Variable(tf.truncated_normal([nInputs, width], 0.0, 0.01, dtype=tf.float32))] #list of weight matrices for each lstm layer
		b = [tf.Variable(np.zeros(width), dtype = tf.float32)]

		for i in range(len(arc)-3): #adds the remaining parameters to the lists
			w.append(tf.Variable(tf.truncated_normal([width, width], 0.0, 0.01, dtype=tf.float32))) 
			b.append(tf.Variable(np.zeros(width), dtype = tf.float32))
		rnn_w = [tf.Variable(np.identity(width)*alpha, dtype = tf.float32)]*rnn_layers #list of state weight matrices which propogate through time


		def condition(i,n,w,h1,s,a):#index, memory, state weight matrix, hidden transform, rnn state, temp array
			return tf.less(i,n)

		def step(i,n,w,h1,s,a):
			s = tf.nn.relu(tf.add(tf.matmul(s, w), h1[i]))
			a = a.write(i,s)
			return [i+1, n,w,h1,s,a]
		layer_state = x;

		for i in range(rnn_layers):
			hidden = tf.reshape(layer_state, [-1,arc[i]]) #[a,b,c]->[a*b, c]
			hidden = tf.add(tf.matmul(hidden,w[i]), b[i]) 
			hidden = tf.reshape(hidden, [self.memory, batch_size, width])

			temp = tf.TensorArray(dtype=tf.float32, size=self.memory) #array which holds state for each time step
			state = tf.zeros([batch_size, width], dtype=tf.float32)# initial state


			index = 0
			_,_,_,_,_,out = tf.while_loop(condition, step, [index, memory, rnn_w[i],hidden, state, temp])

			out = out.stack()
			layer_state = out




		w2 = tf.Variable(np.random.rand(width, nOutputs), dtype=tf.float32)
		b2 = tf.Variable(np.zeros(nOutputs), dtype = tf.float32)
		out = layer_state[-1]
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

