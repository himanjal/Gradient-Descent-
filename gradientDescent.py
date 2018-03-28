"""
Gradient Descent

Author:
Himanjal Sharma

METHOD 1
Mean Squared Errors
Training Set: 0.0494021194165
Testing Set: 0.0922196972584

METHOD 2
Mean Squared Errors
Training Set: 0.0499311452762
Testing Set: 0.0899913231001

METHOD 3
Mean Squared Errors
Training Set: 0.0499158922151
Testing Set: 0.0898714689298

"""




import numpy as np


def convertX(faces):

	newX = np.zeros((faces.shape[0],24*24 +1))

	for i in range(0, faces.shape[0]):
		imgvec = faces[i:i+1,0,:]
		for j in range(1,24):
			imgvec = np.hstack((imgvec, faces[i:i+1,j,:]))
		b = np.array([[1]])
		imgvec = np.hstack((imgvec,b))
		newX[i,:] = imgvec 

	return newX.T

def calculateWeights(faces, labels):

	X = faces

	p1 = np.dot(X,X.T)
	p2 = np.dot(X,labels)
	W = np.linalg.solve(p1,p2)

	return W


def calculateYhat(faces, weights):

	X = faces
	return np.dot(X.T,weights)


def calculateFmse(y, yHat):

	return (np.mean((yHat - y)**2))/2


def calculateRegularizedFmse(Y, yHat, weights, alpha):

	#Seperating inputs into data and bias
	bindex = Y.shape[0] - 1

	Ydata = Y[0:bindex]
	Yb = Y[bindex]

	yHatData = yHat[0:bindex]
	yHatB = yHat[bindex]

	bindex = weights.shape[0] - 1
	weightsData = weights[0:bindex]
	weightsB = weights[bindex]


	p1 = calculateFmse(Ydata, yHatData)
	p2 = (alpha/2) * np.dot(weightsData.T, weightsData)

	result1 = p1 + p2
	result2 = (np.mean((yHatB - Yb)**2))/2

	return (576 * result1 + result2)/577


def getGradient(faces, labels, weights):

	X = faces

	p1 = X/X.shape[0]
	p2 = np.dot(X.T, weights)
	p3 = p2 - labels

	return np.dot(p1,p3)


def performGradientDescent(faces, labels, learningRate, tolerance):

	#picking random weight
	w = 0.01 * np.random.randn(576)

	weights = w

	oldFmse = 0.0
	yHat = calculateYhat(faces, weights)
	newFmse = calculateFmse(labels, yHat)

	while(abs(oldFmse - newFmse) > tolerance):

		gradient = getGradient(faces, labels, weights)
		#print "gradient" , gradient
		weights = weights - learningRate * gradient
		yHat = calculateYhat(faces, weights)
		oldFmse = newFmse
		newFmse = calculateFmse(labels, yHat)
		#print oldFmse - newFmse

	return weights


def performRegularizedGD(faces, labels, learningRate, tolerance):

	#picking random weight
	w = 0.01 * np.random.randn(576)

	weights = w

	oldFmse = 0
	yHat = calculateYhat(faces, weights)
	newFmse = calculateRegularizedFmse(labels, yHat, weights, 1)

	while(tolerance < abs(oldFmse - newFmse)):

		gradient = getGradient(faces, labels, weights)
		weights = weights - learningRate * gradient
		yHat = calculateYhat(faces, weights)
		oldFmse = newFmse
		newFmse = calculateRegularizedFmse(labels, yHat, weights, 1)

	return weights


def doHomework2(trainingFaces,trainingLabels,testingFaces,testingLabels):

	log("Machine Learning\tHomework 2\n")
	
	#Method 1
	log("METHOD 1")
	X = convertX(trainingFaces)
	weights = calculateWeights(X, trainingLabels)
	yHat = calculateYhat(X, weights)
	fmse = calculateFmse(trainingLabels, yHat)

	log("Mean Squared Errors")
	log("Training Set: {}".format(fmse))

	X = convertX(testingFaces)
	yHat = calculateYhat(X, weights)
	fmse = calculateFmse(testingLabels, yHat)

	log("Testing Set: {}\n".format(fmse))


	#Method 2
	log("METHOD 2")
	X = convertX(trainingFaces)
	Y = trainingLabels
	learningRate = 0.001
	tolerance = 0.000001
	weights = performGradientDescent(X, Y, learningRate, tolerance)

	yHat = calculateYhat(X, weights)
	fmse = calculateFmse(Y, yHat)

	log("Mean Squared Errors")
	log("Training Set: {}".format(fmse))

	X = convertX(testingFaces)
	Y = testingLabels
	yHat = calculateYhat(X, weights)
	fmse = calculateFmse(Y, yHat)

	log("Testing Set: {}\n".format(fmse))


	#Method 3
	log("METHOD 3")
	X = convertX(trainingFaces)
	Y = trainingLabels
	learningRate = 0.001
	tolerance = 0.000001
	weights = performRegularizedGD(X, Y, learningRate, tolerance)

	yHat = calculateYhat(X, weights)
	fmse = calculateFmse(Y, yHat)

	log("Mean Squared Errors")
	log("Training Set: {}".format(fmse))

	X = convertX(testingFaces)
	Y = testingLabels
	yHat = calculateYhat(X, weights)
	fmse = calculateFmse(Y, yHat)

	log("Testing Set: {}".format(fmse))


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


def log(text):
	global logString
	print text
	text = text + "\n"
	logString = logString + text 
	


logString = ""
logging = False

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    doHomework2(trainingFaces,trainingLabels,testingFaces,testingLabels)
    if logging:
    	with open("LogFile.txt", 'w') as fh:
    		fh.write(logString)
    		fh.close()
