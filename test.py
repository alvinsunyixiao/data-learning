from scipy.optimize import minimize
import scipy.io as sio
import numpy as np
import func
boxData = sio.loadmat('boxData')
X = boxData['X']
y = boxData['y']
initpm = func.initNNparams(1200,100,100,4)
costFunc = func.costFunction
result = minimize(costFunc,initpm,args=(1200,100,100,4,X,y,0.0),method='Newton-CG',jac=True,options={'disp':True,'maxiter':100})
print result