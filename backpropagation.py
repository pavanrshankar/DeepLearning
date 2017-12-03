import numpy as np
import matplotlib.pyplot as plt


##################################################################
#Backpropagation implementation using 1 hidden layer
##################################################################

#Get initial data
def getInitialDataWeights():
	X1 = np.random.randn(500,2) + np.array([10,-2])
	X2 = np.random.randn(500,2) + np.array([2,10])
	X3 = np.random.randn(500,2) + np.array([-2,2])

	X = np.vstack([X1,X2,X3])
	Y = np.array([0]*500 + [1]*500 + [2]*500)
	no_classes = 3

	return X,Y,no_classes

#ANN architecture
def initneuralNetwork():
	firstLayer = 2
	secLayer = 10
	thirdLayer = 3

	W1 = np.random.randn(firstLayer, secLayer)
	b1 = np.random.randn(1, secLayer)
	W2 = np.random.randn(secLayer, thirdLayer)
	b2 = np.random.randn(1, thirdLayer)

	return W1,b1,W2,b2

#Compares element by element and takes rows that are True
def getaccuracy(Y, P):
	res = (np.array(Y) == np.array(P)).tolist()
	return res.count(True) / float(len(res))

#Encoding for multi-class classification
def multiClassEncoding(v, no_classes):
	N = v.shape[0]
	v_enc = np.zeros((N, no_classes))
	v_enc[np.arange(N), v] = 1
	return v_enc

#Softmax for multiclass classification
def feedforward(X, W1, W2, b1, b2):
	Z1 = 1 / (1 + np.exp(-X.dot(W1) - b1))
	Z2 = Z1.dot(W2) + b2
	expZ2 = np.exp(Z2) 
	P = expZ2 / expZ2.sum(axis=1, keepdims=True)
	return Z1, np.argmax(P, axis=1)

#Vectorized Backpropagation for Speed
def backpropagation(X, Y, W1, W2, b1, b2, no_classes):
	T = multiClassEncoding(Y, no_classes)
	
	acc = []
	alpha = 0.00001

	for epoch in range(10000):
		#Predict using current weights
		Z1, P = feedforward(X, W1, W2, b1, b2)
		
		#Accuracy using current weights
		a = getaccuracy(Y, P)
		print(a)
		acc += [a]

		#Encoding for multi-class
		P = multiClassEncoding(P, no_classes)
		
		#Changing weights and constants	
		tempW2 = W2 + alpha * (Z1.T).dot(T-P) 
		b2 = b2 + alpha * (T-P).sum(axis=0, keepdims=True)
		W1 = W1 + alpha * ((X.T).dot((T-P).dot(W2.T) * Z1 * (1-Z1))).sum(axis=0, keepdims=True)
		b1 = b1 + alpha * ((T-P).dot(W2.T) * Z1 * (1-Z1)).sum(axis=0, keepdims=True)
		W2 = tempW2
		
	plt.plot(acc)
	plt.show()	
				
if __name__ == '__main__':
	X,Y,no_classes = getInitialDataWeights()
	W1,b1,W2,b2 = initneuralNetwork()
	backpropagation(X, Y, W1, W2, b1, b2, no_classes)

