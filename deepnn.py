import numpy as np
import tensorflow as tf

#Tensorflow basics
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

#Need to create placeholders for variable before usage in sessions
v = tf.placeholder(tf.float32)
A = tf.placeholder(tf.float32)
w = tf.matmul(A,v)

with tf.Session() as session:
	output = session.run(w, feed_dict={A:np.random.rand(5,5),v:np.random.rand(5,1)})
	print(output, type(output))

#Need to create placeholders and initialize variables before usage
u = tf.Variable(20.0)
cost = u*u + u + 1
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(init)
	for i in range(200):
		session.run(train_op)
		print("i:%d, cost:%f"%(i, cost.eval()))
