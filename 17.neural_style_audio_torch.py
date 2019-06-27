import tensorflow as tf
import librosa  # used to abstract voice file
import numpy as np
import os
#import shlex  # python2 pipes
 
# audio file path
content_audio = "./Traveling_Light.mp3"
style_audio = "./东风破.mp3"

# for  add Jay Chou's flavor for music  <Traveling Light> 
 
#  get one clip of audio,default get the beginning 10s length, otherwise needs more memory
def cut_audio(filename, start_pos='00:00:00', lens=10):
	newfile = os.path.splitext(os.path.basename(filename))[0] + '_' + str(lens) + 's.mp3'

	# make sure ffmpeg installed in system
	cmd = "ffmpeg -i {} -ss {} -t {} {}".format(filename, start_pos, lens, newfile)
	os.system(cmd)
	return newfile
 
content_audio_10s = cut_audio(content_audio, start_pos='00:00:33')
style_audio_10s = cut_audio(style_audio, start_pos='00:00:38')
 
# Short Time Fourier Transform audio to spectrogram（convert 1 dimensional signal to 2 dimensional , seem as image）
# https://en.wikipedia.org/wiki/Short-time_Fourier_transform
N_FFT = 2048
def read_audio(filename):
	x, fs = librosa.load(filename)
	S = librosa.stft(x, N_FFT)
	p = np.angle(S)
 
	S = np.log1p(np.abs(S[:,:430]))
	return S, fs
 
content_data, _ = read_audio(content_audio_10s)
style_data, fs = read_audio(style_audio_10s)
 
samples_n = content_data.shape[1]  # 430
channels_n = style_data.shape[0]   # 1025
 
style_data = style_data[:channels_n, :samples_n]
 
content_data_tf = np.ascontiguousarray(content_data.T[None,None,:,:])
style_data_tf = np.ascontiguousarray(style_data.T[None,None,:,:])
 
# filter shape "[filter_height, filter_width, in_channels, out_channels]"
N_FILTERS = 4096
std = np.sqrt(2) * np.sqrt(2.0 / ((channels_n + N_FILTERS) * 11))
kernel = np.random.randn(1, 11, channels_n, N_FILTERS)*std
 
# content and style features
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
	# data shape "[batch, in_height, in_width, in_channels]",
	x = tf.placeholder('float32', [1, 1, samples_n, channels_n], name="x")
 
	kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
	conv = tf.nn.conv2d(x, kernel_tf, strides=[1, 1, 1, 1], padding="VALID", name="conv")
 
	net = tf.nn.relu(conv)
 
	content_features = net.eval(feed_dict={x: content_data_tf})
	style_features = net.eval(feed_dict={x: style_data_tf})
 
	features = np.reshape(style_features, (-1, N_FILTERS))
	style_gram = np.matmul(features.T, features) / samples_n
 
# Optimize
ALPHA= 0.01   # the greater ALPHA is , the more dorminate the  content is ; if ALPHA = 0, represent there is no content
result = None
with tf.Graph().as_default():
	learning_rate= 0.001
	x = tf.Variable(np.random.randn(1, 1, samples_n, channels_n).astype(np.float32)*learning_rate, name="x")
	kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
	conv = tf.nn.conv2d(x, kernel_tf, strides=[1, 1, 1, 1], padding="VALID", name="conv")
 
	net = tf.nn.relu(conv)
 
	content_loss = ALPHA * 2 * tf.nn.l2_loss(net - content_features)
	style_loss = 0
 
	_, height, width, number = map(lambda i: i.value, net.get_shape())
 
	size = height * width * number
	feats = tf.reshape(net, (-1, number))
	gram = tf.matmul(tf.transpose(feats), feats)  / samples_n
	style_loss = 2 * tf.nn.l2_loss(gram - style_gram)
 
	# loss
	loss = content_loss + style_loss
	opt = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': 300})
 
	# Optimization
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		opt.minimize(sess)
		result = x.eval()
 
# convert spectrogram to wav  audio
audio = np.zeros_like(content_data)
audio[:channels_n,:] = np.exp(result[0,0].T) - 1
 
p = 2 * np.pi * np.random.random_sample(audio.shape) - np.pi
for i in range(500):
	S = audio * np.exp(1j*p)
	x = librosa.istft(S)
	p = np.angle(librosa.stft(x, N_FFT))
 
librosa.output.write_wav("output.mp3", x, fs)
