import numpy as np
import random as random

trainingAmount = 5000	#how many examples it will have to learn
testingAmount = 1000	#how many examples it will guess the output of


#2D array to put all the data in
data = np.zeros((trainingAmount + testingAmount, 6))

#for all sets of input/output for both training and testing amount
for i in range(0, trainingAmount + testingAmount):
	#randomly create input for the first 5 inputs for the ith data set
	for j in range(0, 5):
		data[i][j] = random.randint(0,9)	#possible values are 0 to 9
	"""
	rules for the output after the inputs are randomly chosen:
	if the first input is less than 5, output is always 0.
	if the previous rule doesn't apply, then if the 2nd input is odd, the output will always be 1
	if the rules above don't apply, then if the 3rd or 4th inputs are less than or equal to 4, the output will always be 2
	if the rules above don't apply, then if any of the inputs are 0, then the output will always be 3
	if none of the above rules apply, the output will always be 4.
	"""
	if (data[i][0] < 5):
		data[i][5] = 0
	elif (data[i][1]%2 == 1):
		data[i][5] = 1
	elif (data[i][3] <= 4 or data[i][4] <= 4):
		data[i][5] = 2
	elif (data[i][0]*data[i][1]*data[i][2]*data[i][3]*data[i][4] == 0):
		data[i][5] = 3
	else:
		data[i][5] = 4

#writes data to file 'data.txt'
thefile = open('data.txt', 'w')
for i in range(0, trainingAmount + testingAmount):
	for j in range(0, 5):
		thefile.write("%d," % data[i][j])
	thefile.write("%d\n" % data[i][5])
thefile.close()

#reads data in from file 'data.txt' and organizes it into 4 2D arrays:
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
		else:
			testingInput[i-trainingAmount][j] = currentline[j]
	#the last number in a line is the output
	if i < trainingAmount:
		trainingOutput[i][0] = currentline[5]
	else:
		testingOutput[i-trainingAmount][0] = currentline[5]
	#incremment line count before for loop ends
	i = i + 1
"""
print trainingInput
print trainingOutput
print testingInput
print testingOutput
"""