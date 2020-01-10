import numpy as np
from calculation_cost import *

def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
    '''
	X = Matrix of X with added bias unit
	y = Vector of y
	theta = vector of thetas np.random.randn(j,1)
	learning_rate
	iterations = no of iterations
	
	Returns the final theta vector and array of cost history over no of iterations
	'''
	m = len(y)
	cost_history = np.zeros(iterations)
	theta_history = np.zeros((iterations,2))
	for it in range(iterations):
	    prediction = np.dot(X,theta)
		
		theta = theata - (1/m)*learning_rate*(X.T.dot((prediction - y)))
		theta_history[it,:] = theta.T
		cost_history[it] = cal_cost(theta,X,y)
		
		
	return theta, cost_history, theta_history
		
	

