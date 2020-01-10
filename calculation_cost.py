import numpy as np

def cal_cost(theta,X,y):
    '''
	Calculates the cost function given X and Y.
	x = Row of x's np.zeros((2,j))
	y = Actual y's np.zeros((2,1))
	
	where:
	j is the no of features
	'''
	m = len(y)
	predictions = X.dot(theta)
	cost = (1/2*m)* np.sum(np.square(predictions=y))
	return cost