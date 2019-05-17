from random import choice 
from numpy import array, dot, random
import numpy as np

class Data:
	"""here kept all the training sets and other data"""
	training_data_or = [(array([0,0,1]), 0),(array([0,1,1]), 1),(array([1,0,1]), 1),(array([1,1,1]), 1),]
	training_data_and = [(array([0,0,1]), 0),(array([0,1,1]), 0),(array([1,0,1]), 0),(array([1,1,1]), 1),]
	thethas_and = [-1, 0, 1]
	thethas_or = [-1, 0, 1]
	errors = [] 
	bias = 0.5 
	n = 100

unit_step = lambda x: 0 if x < 0 else 1

def perceptron(x, thethas, bias):
    v = np.dot(thethas, x) + bias
    y = unit_step(v)
    return y

def trainedOR():
	data = Data()
	for i in range(data.n): 
	    x, expected = choice(data.training_data_or) 
	    result = dot(data.thethas_or, x) 
	    data.error = expected - unit_step(result) 
	    data.errors.append(data.error) 
	    data.thethas_or += data.bias * data.error * x

	for x, _ in data.training_data_or: 
	    result = dot(x, data.thethas_or) 
	    print("OR {}: {} -> {}".format(x[:2], result, unit_step(result)))

def trainedAND():
	data = Data()
	for i in range(data.n): 
	    x, expected = choice(data.training_data_and) 
	    result = dot(data.thethas_and, x) 
	    data.error = expected - unit_step(result) 
	    data.errors.append(data.error) 
	    data.thethas_and += data.bias * data.error * x

	for x, _ in data.training_data_and: 
	    result = dot(x, data.thethas_and) 
	    print("AND {}: {} -> {}".format(x[:2], result, unit_step(result)))

def repNOT(x):
	return perceptron(x, thethas=-1, bias=0.5)
	
def perAND(x):
    w = np.array([1, 1])
    bias = -1.5
    return perceptron(x, w, bias)

def repOR(x):
    w = np.array([1, 1])
    bias = -0.5
    return perceptron(x, w, bias)

def perXOR(x):
    new_x = np.array([repNOT(perAND(x)), repOR(x)])
    output = perAND(new_x)
    return output
	
def perXNOR(x):
    new_x = np.array([repNOT(perAND(x)), repOR(x)])
    output = repNOT(perAND(new_x))
    return output

def test_mod():
	try:
	    assert perXOR([1, 1]) == 0, 'passed'
	    assert perXNOR([1, 1]) == 1, 'passed'
	    assert unit_step(-0.5) == 0, 'passed'
	    assert unit_step(0.5) == 1, 'passed'
	    assert unit_step(0.0) == 1, 'passed'
	except AssertionError:
	    raise AssertionError("Some of the tests were failed")
	else:
	    print("All the tests passed\n")

if __name__ == '__main__':
    
    test_mod()

    trainedAND()
    trainedOR()    

    print("xOR [{} {}]: -> {}".format(1, 1, perXOR([1, 1])))
    print("xOR [{} {}]: -> {}".format(1, 0, perXOR([1, 0])))
    print("xOR [{} {}]: -> {}".format(0, 1, perXOR([0, 1])))
    print("xOR [{} {}]: -> {}".format(0, 0, perXOR([0, 0])))

    print("xnOR [{} {}]: -> {}".format(1, 1, perXNOR([1, 1])))
    print("xnOR [{} {}]: -> {}".format(1, 0, perXNOR([1, 0])))
    print("xnOR [{} {}]: -> {}".format(0, 1, perXNOR([0, 1])))
    print("xnOR [{} {}]: -> {}".format(0, 0, perXNOR([0, 0])))