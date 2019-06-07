import tensorflow as tf
import numpy as np
 
# download mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot=True)
 
 
#a img is 28*28,FNN input the whole image at one time, RNN separate it several parts
chunk_size = 28
chunk_n = 28
 
rnn_size = 256
 
n_output_layer = 10   # output layer size
 
X = tf.placeholder('float', [None, chunk_n, chunk_size]) 
Y = tf.placeholder('float')

# define the network
def recurrent_neural_network(data):
	layer = {'w_':tf.Variable(tf.random_normal([rnn_size, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}
 
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
 
	data = tf.transpose(data, [1,0,2])
	data = tf.reshape(data, [-1, chunk_size])
	data = tf.split(0, chunk_n, data)
	outputs, status = tf.nn.rnn(lstm_cell, data, dtype=tf.float32)
 
	ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
 
	return ouput
 
# use 100 items of data a time 
batch_size = 100
 
# training
def train_neural_network(X, Y):
	predict = recurrent_neural_network(X)
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost_func)
 
	epochs = 13
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		epoch_loss = 0
		for epoch in range(epochs):
			for i in range( int(mnist.train.num_examples/batch_size) ):
				x, y = mnist.train.next_batch(batch_size)
				x = x.reshape([batch_size, chunk_n, chunk_size])
				_, c = session.run([optimizer, cost_func], feed_dict={X:x,Y:y})
				epoch_loss += c
			print(epoch, ' : ', epoch_loss)
 
		correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('accurate: ', accuracy.eval({X:mnist.test.images.reshape(-1, chunk_n, chunk_size), Y:mnist.test.labels}))
 
train_neural_network(X,Y)
