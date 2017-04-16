#%%
import numpy as np


class NeuralNetwork:
    def __init__(self,inputSize,hiddenSize,outputSize,learningRate):

        self.inputLayer = inputSize  
        self.hiddenLayer = hiddenSize  
        self.outputLayer = outputSize
        self.learningRate = learningRate

        self.V = 0.01*np.random.randn(self.inputLayer + 1, self.hiddenLayer)  
        self.W = 0.01*np.random.randn(self.hiddenLayer + 1, self.outputLayer) 


    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        return 1 - np.tanh(z)**2

    def feedForward(self, X):

        self.z2 = np.ravel(np.dot(self.V.T, X.reshape(X.shape[0], 1)))

        self.a2 = np.append(self.tanh(self.z2),1)

        self.z3 = np.ravel(np.dot(self.W.T, self.a2.reshape(self.a2.shape[0], 1)))

        yHat = self.sigmoid(self.z3)
        return yHat

    def crossEntropyError(self, X, y):
        self.yHat = self.feedForward(X)
        J = sum(-(np.multiply(y,np.log(self.yHat)) + np.multiply(1.0-y, np.log(1.0-self.yHat))))
        return J

    def crossEntropyErrorD(self, X, y):
        self.yHat = self.feedForward(X)
        d3 = self.yHat-y

        dJdW = np.dot(self.a2.reshape(self.a2.shape[0], 1), d3.reshape(d3.shape[0],1).T)

        d2a = np.dot(self.W, d3)
        d2b = self.tanh_prime(self.z2)
        d2a = np.delete(d2a, 200)
        d2 = np.multiply(d2a, d2b)

        dJdV = np.dot(X.reshape(X.shape[0], 1), d2.reshape(d2.shape[0], 1).T)
        
        return dJdV, dJdW

    def crossEntropyGradients(self, X, y):
        dJdV, dJdW = self.crossEntropyErrorD(X, y)
        self.crossEntropyError(X, y)
        return dJdV, dJdW


    def trainNetwork(self, X, y):
        gradient_V, gradient_W = self.crossEntropyGradients(X, y)
        self.W -= self.learningRate * gradient_W
        self.V -= self.learningRate * gradient_V

    def predict(self, X):
        return self.feedForward(X)