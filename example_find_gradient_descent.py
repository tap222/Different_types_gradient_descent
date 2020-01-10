from gradient_descent import *

lr = 0.01
n_iter = 1000

X_b = np.random.randn(2,1)

X_b = np.c_[np.ones((len(X),1)),X]
theta, cost_history, theta_history = gradient_descent(X_b,y,theta,lr,n_iter)

print('Theta0: {:0.3f},\nTheta1:{:0.3f}'.format(theta[0][0],theta[1][0]))
print('Final cost/MSE: {:0.3f}'.format(cost_history[-1]))