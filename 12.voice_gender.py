import os
import requests
import pandas as pd
import numpy as np
import random
import tensorflow as tf  # 0.12
from sklearn.model_selection import train_test_split
 
# download dataset
if not os.path.exists('voice.csv'):
	url = './voice.csv'  # find in this folder
	data = requests.get(url).content
	with open('voice.csv', 'wb') as f:
		f.write(data)
 
voice_data = pd.read_csv('voice.csv')
#print(voice_data.head())
#print(voice_data.tail())
 
voice_data = voice_data.values
# get voice property and classification labes
voices = voice_data[:, :-1]
labels = voice_data[:, -1:]  #  ['male']  ['female']
 
# change classification to one-hot vector
labels_tmp = []
for label in labels:
	tmp = []
	if label[0] == 'male':
		tmp = [1.0, 0.0]
	else:  # 'female'
		tmp = [0.0, 1.0]
	labels_tmp.append(tmp)
labels = np.array(labels_tmp)
 
# shuffle
voices_tmp = []
lables_tmp = []
index_shuf = [i for i in range(len(voices))]
random.shuffle(index_shuf)
for i in index_shuf:
    voices_tmp.append(voices[i])
    lables_tmp.append(labels[i])
voices = np.array(voices_tmp)
labels = np.array(lables_tmp)
 
train_x, test_x, train_y, test_y = train_test_split(voices, labels, test_size=0.1)
 
banch_size = 64
n_banch = len(train_x) // banch_size
 
X = tf.placeholder(dtype=tf.float32, shape=[None, voices.shape[-1]])  # 20
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])
 
# 3 layers（feed-forward）
def neural_network():
	w1 = tf.Variable(tf.random_normal([voices.shape[-1], 512], stddev=0.5))
	b1 = tf.Variable(tf.random_normal([512]))
	output = tf.matmul(X, w1) + b1
	
	w2 = tf.Variable(tf.random_normal([512, 1024],stddev=.5))
	b2 = tf.Variable(tf.random_normal([1024]))
	output = tf.nn.softmax(tf.matmul(output, w2) + b2)
 
	w3 = tf.Variable(tf.random_normal([1024, 2],stddev=.5))
	b3 = tf.Variable(tf.random_normal([2]))
	output = tf.nn.softmax(tf.matmul(output, w3) + b3)
	return output
 
# training
def train_neural_network():
	output = neural_network()
 
	cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, Y)))
	lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
	opt = tf.train.AdamOptimizer(learning_rate=lr)
	var_list = [t for t in tf.trainable_variables()]
	train_step = opt.minimize(cost, var_list=var_list)
 
	#saver = tf.train.Saver(tf.global_variables())
	#saver.restore(sess, tf.train.latest_checkpoint('.'))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#summary_writer = tf.train.SummaryWriter('voices')
		for epoch in range(200):
			sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))
 
			for banch in range(n_banch):
				voice_banch = train_x[banch*banch_size:(banch+1)*(banch_size)]
				label_banch = train_y[banch*banch_size:(banch+1)*(banch_size)]
				_, loss = sess.run([train_step, cost], feed_dict={X: voice_banch, Y: label_banch})
				print(epoch, banch, loss)
 
		# accuracy
		prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
		accuracy = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
		print("accuracy", accuracy)
 
		#prediction = sess.run(output, feed_dict={X: test_x})
 
train_neural_network()
