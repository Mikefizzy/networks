from networks import IRNN
from networks import LSTM
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def f(x):
	return np.cos(x)
n_samples = 1000
valSize = 200;
x_values = np.linspace(0,4*np.pi, n_samples)
y_values = f(x_values)

x_data = []
y_data = []
memory = 100;
for i in range(n_samples-memory):
	x_data.append(y_values[i:i+memory])
	y_data.append(y_values[i+memory])
batch_size = n_samples-memory
x_data = np.reshape(np.array(x_data), [batch_size, memory, 1])
y_data = np.reshape(np.array(y_data), [batch_size, 1])
endpoint = n_samples-valSize
train_x = x_data[:endpoint]
train_y = y_data[:endpoint]
val_x = x_data[endpoint:]
val_y = y_data[endpoint:]
net = LSTM([1,1000,1000,1], memory, activation = 'none', cost = 'mean_square')
with tf.Session() as sess:
	costs = []
	sess.run(tf.global_variables_initializer())
	for i in range(30):
		cost = sess.run(net.cost, feed_dict = {net.x: val_x, net.y:val_y})
		sess.run(net.optimizer,feed_dict = {net.x:train_x, net.y:train_y})
		costs.append(cost)
		print(cost)
	plt.plot(costs)
	plt.show()