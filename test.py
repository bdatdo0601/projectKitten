from NeuralNetwork import NeuralNetwork
# from videoSupport import *
from Trainer import trainer
from NeuralNetwork import computeNumericalGradient
import numpy as np

NN = NeuralNetwork()
#input training dataset
#reads data in from file 'data.txt' and organizes it into 4 2D arrays:
trainingAmount = 1000	#how many examples it will have to learn
testingAmount = 100	#how many examples it will guess the output of
trainingInput = np.zeros((trainingAmount, 5))
trainingOutput = np.zeros((trainingAmount, 1))
testingInput = np.zeros((testingAmount, 5))
testingOutput = np.zeros((testingAmount, 1))
file = open('data2.txt', 'r')
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
	#incremment line count before for loop ends
	i = i + 1
    # if (i > (trainingAmount+testingAmount)):
    #     break

# print trainingInput
# print trainingOutput
# print testingInput
# print testingOutput

#normalize dataset
X = testingInput
# trainingOutput = trainingOutput

#Train data
T = trainer(NN)
T.train(X, testingOutput)


#Testing


# X = np.array(([6,5,1,8,1], [6,9,4,7,9], [0,1,6,3,1]), dtype=float)
# y = np.array(([1], [1], [0]), dtype=float)
# X = X/9
# y = y/4
print(NN.forward(testingInput))
print(testingOutput)


numgrad = computeNumericalGradient(NN, X, testingOutput)

grad = NN.computeGradients(X, testingOutput)

gradErr = np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

print (numgrad)
print (grad)
print (gradErr)
