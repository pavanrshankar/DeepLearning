import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
####################################NOTES##############################################
Accuracy: 98%

Data: (input, label) -> ([784 pixel vector], [10 class result vector])

Description: Input is fed into NN in batch size of 100 for 10 epochs where each epoch
covers entire data set. Weights are randomly set using truncated_normal with stddev
of 0.1 to ensure small initial weights as large weights tend to change behaviour of 
output drastically. All selected weights beyond 2 steps of stddev from mean are discarded 
and reselected. AdamOptimizer uses small steps of 0.001 and weights are pushed to change 
due to small biases that fire up the neurons, thereby contributing to result and in turn 
allowing backpropagation algorithm to change weight of nodes that influenced the 
result(Inactive node weights remain unaltered)
#######################################################################################
'''

#Reads data into path specified with label having one active value
mnsit = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl = 1500
n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_layer = {'weights':tf.Variable(tf.truncated_normal([784, n_nodes_hl],stddev=0.1)),'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_hl]))}
	output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl, n_classes],stddev=0.1)),'biases':tf.Variable(tf.constant(0.1,shape=[n_classes]))}				  

	l1 = tf.add(tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])				  
	l1 = tf.nn.relu(l1)

	output = tf.add(tf.matmul(l1, output_layer['weights']), output_layer['biases'])				  

	return output

def train_neural_network():
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))		
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	total_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(total_epochs):
			epoch_loss = 0
			for _ in range(int(mnsit.train.num_examples/batch_size)):
				a, b = mnsit.train.next_batch(batch_size)
				print(a.shape)
				_, c = sess.run([optimizer, cost], feed_dict = {x: a, y: b})
				epoch_loss += c
			print('Epoch ', epoch, ' completed out of ', total_epochs, ' loss:', epoch_loss)
			
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))			
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))	
		print('Accuracy:',accuracy.eval({x:mnsit.test.images, y:mnsit.test.labels}))	

train_neural_network()
