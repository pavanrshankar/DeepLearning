import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#####################Backpropagation using tensorflow#######################
''' 
	#Increasing number of hidden layers takes more iterations to converge
	#Decreasng number of hidden nodes takes more iterations to converge
'''
############################################################################
def createDataPoints():
	N = 100
	D = 2
	M1 = 3
	M2 = 3
	K = 3

	X1 = np.random.randn(N,D) + np.array([2,2])
	X2 = np.random.randn(N,D) + np.array([-1,-1])
	X3 = np.random.randn(N,D) + np.array([10,10])
	X = np.vstack([X1, X2, X3])
	Y = [0]*len(X1) + [1]*len(X2) + [2]*len(X3)

	#plt.scatter(X[:,0], X[:,1], c=Y)
	#plt.show()

	T = np.zeros((K*N, K))
	T[np.arange(K*N), Y] = 1

	return D, K, M1, M2, X, T, Y 

def forward(X, W1, b1, W2, b2, W3, b3):
	O1 = tf.nn.sigmoid(tf.matmul(X,W1)+b1)
	O2 = tf.nn.sigmoid(tf.matmul(O1,W2)+b2)
	O3 = tf.matmul(O2,W3)+b3
	return O3

def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def backpropagation(D, K, M1, M2, X, T, Y):
	#init_weights gives variables to manipulate
	W1 = init_weights([D, M1])
	b1 = init_weights([1, M1])
	W2 = init_weights([M1, M2])
	b2 = init_weights([1, M2])
	W3 = init_weights([M2, K])
	b3 = init_weights([1, K])	
	
	#Recommended to convert numpy objects to tf as type mismatch can occur
	X = tf.convert_to_tensor(X, np.float32)
	pred_x = forward(X, W1, b1, W2, b2, W3, b3)
	predict_op = tf.argmax(pred_x, 1)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_x, labels=T))
	train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
	
	init_op = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init_op)

		for i in range(100000):
			sess.run(train_op)
			pred = sess.run(predict_op)
			if i % 10 == 0:
				print("i:%d, cost:%f"%(i,np.mean(Y == pred)))

if __name__ == '__main__':
	D, K, M1, M2, X, T, Y = createDataPoints()
	backpropagation(D, K, M1, M2, X, T, Y)

