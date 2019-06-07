import tensorflow as tf   
from sklearn.preprocessing import scale  
import pandas as pd
import numpy as np
 
training_data = pd.read_csv("trainingData.csv",header = 0)
# print(training_data.head())
train_x = scale(np.asarray(training_data.ix[:,0:520]))
train_y = np.asarray(training_data["BUILDINGID"].map(str) + training_data["FLOOR"].map(str))
train_y = np.asarray(pd.get_dummies(train_y))
 
test_dataset = pd.read_csv("validationData.csv",header = 0)
test_x = scale(np.asarray(test_dataset.ix[:,0:520]))
test_y = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
test_y = np.asarray(pd.get_dummies(test_y))
 
output = train_y.shape[1]
X = tf.placeholder(tf.float32, shape=[None, 520])  # input
Y = tf.placeholder(tf.float32,[None, output]) # output

# define neural network
def neural_networks():
	# --------------------- Encoder -------------------- #
	e_w_1 = tf.Variable(tf.truncated_normal([520, 256], stddev = 0.1))
	e_b_1 = tf.Variable(tf.constant(0.0, shape=[256]))
	e_w_2 = tf.Variable(tf.truncated_normal([256, 128], stddev = 0.1))
	e_b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
	e_w_3 = tf.Variable(tf.truncated_normal([128, 64], stddev = 0.1))
	e_b_3 = tf.Variable(tf.constant(0.0, shape=[64]))
	# --------------------- Decoder  ------------------- #
	d_w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev = 0.1))
	d_b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
	d_w_2 = tf.Variable(tf.truncated_normal([128, 256], stddev = 0.1))
	d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))
	d_w_3 = tf.Variable(tf.truncated_normal([256, 520], stddev = 0.1))
	d_b_3 = tf.Variable(tf.constant(0.0, shape=[520]))
	# --------------------- DNN  ------------------- #
	w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev = 0.1))
	b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
	w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev = 0.1))
	b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
	w_3 = tf.Variable(tf.truncated_normal([128, output], stddev = 0.1))
	b_3 = tf.Variable(tf.constant(0.0, shape=[output]))
	#########################################################
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(X,       e_w_1), e_b_1))
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, e_w_2), e_b_2))
	encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2, e_w_3), e_b_3))
	layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded, d_w_1), d_b_1))
	layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, d_w_2), d_b_2))
	decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5, d_w_3), d_b_3))
	layer_7 = tf.nn.tanh(tf.add(tf.matmul(encoded, w_1),   b_1))
	layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7, w_2),   b_2))
	out = tf.nn.softmax(tf.add(tf.matmul( layer_8, w_3),   b_3))
	return (decoded, out)
 
# training
def train_neural_networks():
	decoded, predict_output = neural_networks()
 
	us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
	s_cost_function = -tf.reduce_sum(Y * tf.log(predict_output))
	us_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(us_cost_function)
	s_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(s_cost_function)
 
	correct_prediction = tf.equal(tf.argmax(predict_output, 1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
	training_epochs = 20
	batch_size = 10
	total_batches = training_data.shape[0]
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
    
		# ------------ Training Autoencoders - Unsupervised Learning ----------- #
		# autoencoder is a unsupervised learning algorithm, it use the back-propagation algorithm，make the output value equals the input
		for epoch in range(training_epochs):
			epoch_costs = np.empty(0)
			for b in range(total_batches):
				offset = (b * batch_size) % (train_x.shape[0] - batch_size)
				batch_x = train_x[offset:(offset + batch_size), :]
				_, c = sess.run([us_optimizer, us_cost_function],feed_dict={X: batch_x})
				epoch_costs = np.append(epoch_costs, c)
			print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs))
		print("------------------------------------------------------------------")
 
		# ---------------- Training NN - Supervised Learning ------------------ #
		for epoch in range(training_epochs):
			epoch_costs = np.empty(0)
			for b in range(total_batches):
				offset = (b * batch_size) % (train_x.shape[0] - batch_size)
				batch_x = train_x[offset:(offset + batch_size), :]
				batch_y = train_y[offset:(offset + batch_size), :]
				_, c = sess.run([s_optimizer, s_cost_function],feed_dict={X: batch_x, Y : batch_y})
				epoch_costs = np.append(epoch_costs,c)
 
			accuracy_in_train_set = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
			accuracy_in_test_set = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
			print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs)," Accuracy: ", accuracy_in_train_set, ' ', accuracy_in_test_set)
 
 
train_neural_networks()
