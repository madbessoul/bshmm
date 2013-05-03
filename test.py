import hmm
import numpy as np

meth_data = np.genfromtxt('meth.dat', 
	usecols = (0, 2, 4),
	dtype=[('pos','i8'),('cov','i8'), ('count','f8')])[0:5000]


initial = 1./10 * np.array([1] * 10, dtype=np.float32)

states = np.arange(.1, 1, .1)
states = np.append(states, .999)

transition = np.random.rand(100).reshape([10, 10])
for i in xrange(0, len(transition)):
	transition[i,:] *= 1. / sum(transition[i,:])

test_transition = np.eye(10) + .1
for i in xrange(len(test_transition)):
	test_transition[i] /= sum(test_transition[i])

emission = 1./9 * np.ones([4, 10], dtype=np.float32)
observations = np.array([0, 1, 2, 3, 4])


test_data = np.array([0, 0, 0, 2, 2, 3, 1, 2, 3, 2, 0, 0, 1, 2, 2, 3, 3, 3, 3, 1])
test_cov = np.array([1, 3, 5, 3, 5, 8, 6, 2, 3, 4, 8, 12, 10, 8, 6, 22, 4, 6, 8, 10])


# model._forward(test_data)

# counts = meth_data['count'][0:2000]
# coverage = meth_data['cov'][0:2000]

# p, a, scale = model._forward(counts, coverage)
# b = model._backward(counts, coverage, scale)
# ksi = model._ksi(counts, coverage, a, b)
# gamma = model._gamma(a, b)

## CREATE DATA ###
model = hmm.Hmm(states, observations, initial, test_transition)

test_data = np.zeros([5000], dtype=[('cov','i8'), ('count','f8')])
test_data['cov'] = meth_data['cov']

path, test_data['count'] = model.generateData(test_data['cov'], seq_length = 5000)

## VALIDATE PARAMETER EM
val_model = hmm.Hmm(states, observations, initial, transition)
val_model.train(test_data, maxiter=100)