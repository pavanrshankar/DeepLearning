import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Content from:
http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

Notes:
-Tf constructs dependency graphs of variables and evaluates them parallely
-Evaluation is lazy and triggered inside tf.session 
'''

mnsit = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnsit)

