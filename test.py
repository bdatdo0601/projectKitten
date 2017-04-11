from NeuralNetwork import NeuralNetwork
# from videoSupport import *
from Trainer import trainer
from NeuralNetwork import computeNumericalGradient
import numpy as np

NN = NeuralNetwork()
#input training dataset
#reads data in from file 'data.txt' and organizes it into 4 2D arrays:
trainingAmount = 5000	#how many examples it will have to learn
testingAmount = 3	#how many examples it will guess the output of
inputMax = 9
outputMax = 4
trainingInput = np.zeros((trainingAmount, 5))
trainingOutput = np.zeros((trainingAmount, 1))
testingInput = np.zeros((testingAmount, 5))
testingOutput = np.zeros((testingAmount, 1))
file = open('data.txt', 'r')
#if the line number is less than 5000, then the input/outputs will be stored int he training arrays
#otherwise, it'll go into the testing arrays
i = 0
for line in file:
	#data is separated by commas, so this line splits the lines up by comma:
	currentline = line.split(',')
	#first 5 digits are inputs, so they go to the input array
	for j in range(0,5):
		if i < trainingAmount:
			trainingInput[i][j] = currentline[j]
		elif i < (trainingAmount+testingAmount):
			testingInput[i-trainingAmount][j] = currentline[j]
	#the last number in a line is the output
	if i < trainingAmount:
		trainingOutput[i][0] = currentline[5]
	elif i < (trainingAmount+testingAmount):
		testingOutput[i-trainingAmount][0] = currentline[5]
	i = i + 1


#normalize dataset
trainingInput = trainingInput/inputMax
trainingOutput = trainingOutput/outputMax

#Train data
T = trainer(NN)
T.train(trainingInput, trainingOutput)

#Testing
X = np.array(([7,1,6,7,2], [4,9,7,8,6], [9,7,8,2,3]), dtype=float)/inputMax
y = np.array(([1], [0], [1]), dtype=float)/outputMax
yEst = NN.forward(X)
print(yEst)
print(y)
print "relative error: {}".format(abs(y-yEst)/yEst)

#Error estimation
numgrad = computeNumericalGradient(NN, X, y)
grad = NN.computeGradients(X, y)
gradErr = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

# print (gradErr)
