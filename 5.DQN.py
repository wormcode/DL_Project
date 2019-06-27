import pygame
import random
from pygame.locals import *
import numpy as np
from collections import deque
import tensorflow as tf  # http://blog.topspeedsnail.com/archives/10116
import cv2               # http://blog.topspeedsnail.com/archives/4755
 
BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)
 
SCREEN_SIZE = [320,400]
BAR_SIZE = [50, 5]
BALL_SIZE = [15, 15]
 
# output of neural network
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]
 
class Game(object):
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption('Simple Game')
 
		self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
		self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
 
		self.ball_dir_x = -1 # -1 = left 1 = right  
		self.ball_dir_y = -1 # -1 = up   1 = down
		self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
 
		self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
		self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
 
	# action: MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
	# ai control the stick move to right or left back and forth；return the game gui pixel numbers and correspond reward.(pixel-->reward--->reinforcent the stick move toward direction which make most of total rewards
	def step(self, action):
 
		if action == MOVE_LEFT:
			self.bar_pos_x = self.bar_pos_x - 2
		elif action == MOVE_RIGHT:
			self.bar_pos_x = self.bar_pos_x + 2
		else:
			pass
		if self.bar_pos_x < 0:
			self.bar_pos_x = 0
		if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
			self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
			
		self.screen.fill(BLACK)
		self.bar_pos.left = self.bar_pos_x
		pygame.draw.rect(self.screen, WHITE, self.bar_pos)
 
		self.ball_pos.left += self.ball_dir_x * 2
		self.ball_pos.bottom += self.ball_dir_y * 3
		pygame.draw.rect(self.screen, WHITE, self.ball_pos)
 
		if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]+1):
			self.ball_dir_y = self.ball_dir_y * -1
		if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
			self.ball_dir_x = self.ball_dir_x * -1
 
		reward = 0
		if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
			reward = 1    # if hit reward it
		elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
			reward = -1   # not hit penalize it
 
		# get the pixels of game interface
		screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
		pygame.display.update()
		# return game interface UI pixels and correspond reward
		return reward, screen_image
 
# learning_rate
LEARNING_RATE = 0.99
# update gradient
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# test the observe counts
EXPLORE = 500000 
OBSERVE = 50000
# experience store memory size
REPLAY_MEMORY = 500000
 
BATCH = 100
 
output = 3  # number of neurons in output layer.represent 3 kinds of operations-MOVE_STAY:[1, 0, 0]  MOVE_LEFT:[0, 1, 0]  MOVE_RIGHT:[0, 0, 1]
input_image = tf.placeholder("float", [None, 80, 100, 4])  # pixels of game
action = tf.placeholder("float", [None, output])     # operations
 
# define CNN
def convolutional_neural_network(input_image):
	weights = {'w_conv1':tf.Variable(tf.zeros([8, 8, 4, 32])),
               'w_conv2':tf.Variable(tf.zeros([4, 4, 32, 64])),
               'w_conv3':tf.Variable(tf.zeros([3, 3, 64, 64])),
               'w_fc4':tf.Variable(tf.zeros([3456, 784])),
               'w_out':tf.Variable(tf.zeros([784, output]))}
 
	biases = {'b_conv1':tf.Variable(tf.zeros([32])),
              'b_conv2':tf.Variable(tf.zeros([64])),
              'b_conv3':tf.Variable(tf.zeros([64])),
              'b_fc4':tf.Variable(tf.zeros([784])),
              'b_out':tf.Variable(tf.zeros([output]))}
 
	conv1 = tf.nn.relu(tf.nn.conv2d(input_image, weights['w_conv1'], strides = [1, 4, 4, 1], padding = "VALID") + biases['b_conv1'])
	conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides = [1, 2, 2, 1], padding = "VALID") + biases['b_conv2'])
	conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w_conv3'], strides = [1, 1, 1, 1], padding = "VALID") + biases['b_conv3'])
	conv3_flat = tf.reshape(conv3, [-1, 3456])
	fc4 = tf.nn.relu(tf.matmul(conv3_flat, weights['w_fc4']) + biases['b_fc4'])
 
	output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']
	return output_layer
 
# reinforcement learning for beginners: https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
# training
def train_neural_network(input_image):
	predict_action = convolutional_neural_network(input_image)
 
	argmax = tf.placeholder("float", [None, output])
	gt = tf.placeholder("float", [None])
 
	action = tf.reduce_sum(tf.mul(predict_action, argmax), reduction_indices = 1)
	cost = tf.reduce_mean(tf.square(action - gt))
	optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)
 
	game = Game()
	D = deque()
 
	_, image = game.step(MOVE_STAY)
	#  convert to gray values
	image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
	#  convert to binary values
	ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
	input_image_data = np.stack((image, image, image, image), axis = 2)
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		
		saver = tf.train.Saver()
		
		n = 0
		epsilon = INITIAL_EPSILON
		while True:
			action_t = predict_action.eval(feed_dict = {input_image : [input_image_data]})[0]
 
			argmax_t = np.zeros([output], dtype=np.int)
			if(random.random() <= INITIAL_EPSILON):
				maxIndex = random.randrange(output)
			else:
				maxIndex = np.argmax(action_t)
			argmax_t[maxIndex] = 1
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
 
			#for event in pygame.event.get():  macOS needs event circle otherwise white screen
			#	if event.type == QUIT:
			#		pygame.quit()
			#		sys.exit()
			reward, image = game.step(list(argmax_t))
 
			image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
			ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
			image = np.reshape(image, (80, 100, 1))
			input_image_data1 = np.append(image, input_image_data[:, :, 0:3], axis = 2)
 
			D.append((input_image_data, argmax_t, reward, input_image_data1))
 
			if len(D) > REPLAY_MEMORY:
				D.popleft()
 
			if n > OBSERVE:
				minibatch = random.sample(D, BATCH)
				input_image_data_batch = [d[0] for d in minibatch]
				argmax_batch = [d[1] for d in minibatch]
				reward_batch = [d[2] for d in minibatch]
				input_image_data1_batch = [d[3] for d in minibatch]
 
				gt_batch = []
 
				out_batch = predict_action.eval(feed_dict = {input_image : input_image_data1_batch})
 
				for i in range(0, len(minibatch)):
					gt_batch.append(reward_batch[i] + LEARNING_RATE * np.max(out_batch[i]))
 
				optimizer.run(feed_dict = {gt : gt_batch, argmax : argmax_batch, input_image : input_image_data_batch})
 
			input_image_data = input_image_data1
			n = n+1
 
			if n % 10000 == 0:
				saver.save(sess, 'game.cpk', global_step = n)  # persistent the model
 
			print(n, "epsilon:", epsilon, " " ,"action:", maxIndex, " " ,"reward:", reward)
 
 
train_neural_network(input_image)
