	
output = joke_cnn()
predict = tf.argmax(output, 1)
 
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.'))
 
def is_my_face(image):
	res = sess.run(predict, feed_dict={X: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})
	if res[0] == 1:
		return True
	else:
		return False
 
face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
 
while True:
	_, img = cam.read()
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
	for face_x,face_y,face_w,face_h in faces:
		face = img[face_y:face_y+face_h, face_x:face_x+face_w]
 
		face = cv2.resize(face, (64, 64))
 
		print(is_my_face(face))
 
		cv2.imshow('img', face)
		key = cv2.waitKey(30) & 0xff
		if key == 27:
			sys.exit(0)
 
sess.close()
