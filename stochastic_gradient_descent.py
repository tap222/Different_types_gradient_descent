import numpy as np
from calculation_cost import *

def stocashtic_gradient_descent(X,y,theta,learning_rate=0.01,iterations=10):
    '''
	X = matrix of X with added bias unit
	y = Vector of y
	theta = vector of theta np.random.randn(j,1)
	learning_rate
	iterations = no of iterations
	
	Returns the final theta vector and array of cost history over iterations
	'''
	
	m = len(y)
	cost_history = np.zeros(iterations)
	
	for it in range(iterations):
	    cost = 0.0
		for i in range(iterations):
		    rand_ind = np.random.randint(0,m)
			X_i = X[rand_ind,:].reshape(1,X.shape[1])
			y_i = y[rand_ind,:].reshape(1,1)
			predictions = np.dot(X_i,theta)
			
			theta = theta - (1/m)*learning_rate*(X_i.T.dot(prediction - y_i))
			cost += cal_cost(theta,X_i,y_i)
			
		cost_history[it] = cost
		
	return theta, cost_history