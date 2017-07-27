import tensorflow as tf
import numpy as np
from senti_anal_helper import create_feature_sets_and_labels
			
'''
####################################NOTES##############################################
Accuracy: 68%

Data: (input, label) -> ([feature vector], [pos-neg vector])

Description: Input file is scanned and word/feature collection is prepared. Every 
statement is converted to feature vector along with pos([1,0]) or neg([0,1]) label. 
Data is fed into deep neural network with 2 hidden layers and 1500 nodes in each layer. 
#######################################################################################
'''

def neural_network_model(data, inp_size):
	n_nodes_h1 = 1500
	n_nodes_h2 = 1500
	n_classes = 2

	hidden_layer_1 = {'weights':tf.Variable(tf.truncated_normal([inp_size, n_nodes_h1],stddev=0.1)),'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_h1]))}
	hidden_layer_2 = {'weights':tf.Variable(tf.truncated_normal([n_nodes_h1, n_nodes_h2],stddev=0.1)),'biases':tf.Variable(tf.constant(0.1,shape=[n_nodes_h2]))}
	output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_h2, n_classes],stddev=0.1)),'biases':tf.Variable(tf.constant(0.1,shape=[n_classes]))}				  

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])				  
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])				  
	l2 = tf.nn.relu(l2)

	output = tf.add(tf.matmul(l2, output_layer['weights']), output_layer['biases'])				  

	return output

def train_neural_network():	
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

	n_classes = 2
	
	x = tf.placeholder('float', [None, len(train_x[0])])
	y = tf.placeholder('float', [None, n_classes])
	
	prediction = neural_network_model(x, len(train_x[0]))
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))		
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	total_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		batch_size = 100
		for epoch in range(total_epochs):
			epoch_loss = 0
			i = 0
			while i < len(train_x):
				batch_x = train_x[i:i+batch_size]
				batch_y = train_y[i:i+batch_size]
				'''
					(1)x, y takes comma separated array elements/lists
					(2)To store weights across session runs, use tf.Variable
				'''	
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i += batch_size
			print('Epoch ', epoch, ' completed out of ', total_epochs, ' loss:', epoch_loss)
			
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))			
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		return accuracy.eval({x:test_x, y:test_y})

if __name__ == '__main__':
	total_accuracy = 0
	runs = 5

	for i in range(runs):
		accuracy = train_neural_network()
		total_accuracy += accuracy
		print('Accuracy in ', i, ' run is ', accuracy)

	print(total_accuracy/runs);	

