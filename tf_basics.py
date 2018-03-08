import numpy as np
import tensorflow as tf

#####################Tensorflow Theory#######################
'''
	-Lazy evaluation - no evaluation until asked
	-Placeholders are templates using which we can build dependency graphs without data
'''
def variablesUsage():
	x = tf.constant(35, name='x')
	y = tf.Variable(x+5, name='y')

	model = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(model)
		print(sess.run(y))

if __name__ == '__main__':
	variablesUsage()

