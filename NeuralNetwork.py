import numpy as np


"""
Neural Network:
this class will implement a basic neural network system.

inputs to each layer will be represent as a matrix for to simplify calculation.
"""
class NeuralNetwork(object):
    #initialize ANN by taking an list of size for all the hidden layer in between
    def __init__(self, inputSize=5, outputSize=1, hiddenLayersInfo=[2]):
        #amount of inputs
        self.inputSize = inputSize
        #amount of hiddenLayers
        self.hiddenLayers = hiddenLayersInfo
        #amount of outputs
        self.outputSize = outputSize
        #initialize weight
        self.weightAssociated = []
        self.weightAssociated.append(np.random.rand(self.inputSize, self.hiddenLayers[0]))
        for i in range(1, len(self.hiddenLayers)):
            self.weightAssociated.append(np.random.rand(self.hiddenLayers[i-1], self.hiddenLayers[i]))
        self.weightAssociated.append(np.random.rand(self.hiddenLayers[-1], self.outputSize))
        #this is the value at each node as value forward propagate
        self.z = []
        #this will be the the activation funciton value computed from each node
        self.a = []

    #activation function, apply it to scalar, vector, matrix, etc
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    #1st derivative of sigmoid
    def sigmoidP(self, x):
        return np.exp(x)/((1+np.exp(-x))**2)

    #propagate data
    def forward(self, inputs):
        #this is the value at each node as value forward propagate
        self.z = []
        #this will be the the activation funciton value computed from each node
        self.a = []
        if len(inputs[0]) != self.inputSize:
            raise("input not match")
        self.z.append(np.dot(inputs, self.weightAssociated[0]))
        self.a.append(self.sigmoid(self.z[0]))
        for i in range(1, len(self.weightAssociated)):
            self.z.append(np.dot(self.a[i-1], self.weightAssociated[i]))
            self.a.append(self.z[-1])
        return self.a[-1]

    #compute cost (error estimation)
    def costFunction(self, X, y):
        self.yEst = self.forward(X)
        cost = 0.5*sum((y-self.yEst)**2)
        return cost

    #compute deriv of cost respect to each weight for a given training dataset
    #this will return a list of [dJ/dWi] ()
    def costFunctionPrime(self, X, y):
        self.yEst = self.forward(X)
        dJdW = []
        delta = np.multiply(-(y-self.yEst), self.sigmoidP(self.z[-1]))
        dJdW.append(np.dot(self.a[-2].T, delta))
        for i in range(len(self.z)-2, 0, -1):
            delta = np.dot(delta, self.weightAssociated[i+1].T)*self.sigmoidP(self.z[i])
            dJdW = [np.dot(self.a[i].T, delta)] + dJdW
        delta = np.dot(delta, self.weightAssociated[1].T)*self.sigmoidP(self.z[0])
        dJdW = [np.dot(X.T, delta)] + dJdW
        return dJdW

    """
    THIS PART IS FOR CHECKING PURPOSES! REFERENCES SOURCE: https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/partFive.py
    """
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate(tuple(W.ravel() for W in self.weightAssociated))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W_start = 0
        W_end = self.hiddenLayers[0] * self.inputSize
        # print(self.weightAssociated)
        self.weightAssociated[0] = np.reshape(params[W_start:W_end], (self.inputSize , self.hiddenLayers[0]))
        for i in range(1, len(self.weightAssociated)-1):
            W_start = W_end
            W_end = W_end + self.hiddenLayers[i]*self.hiddenLayers[i-1]
            self.weightAssociated[i] = np.reshape(params[W_start:W_end], (self.hiddenLayers[i-1], self.hiddenLayers[i]))
        W_start = W_end
        W_end = W_end + self.hiddenLayers[-1]*self.outputSize
        self.weightAssociated[-1] = np.reshape(params[W_start:W_end], (self.hiddenLayers[-1], self.outputSize))

    def computeGradients(self, X, y):
        DJDW = self.costFunctionPrime(X, y)
        return np.concatenate(tuple(dJdW.ravel() for dJdW in DJDW))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad
