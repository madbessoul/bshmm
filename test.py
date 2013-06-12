import hmm
import numpy as np
import pickle
from pylab import *

# Length of the sequence
L = 1000
meth_data = np.genfromtxt('meth.dat', 
	usecols = (0, 2, 4),
	dtype=[('pos','i8'),('cov','i8'), ('count','f8')])[0:L]


states = np.array([.0001, .1, .2, .3, .4, .5, .6, .7, .8, .9, .9999])

N = len(states)

initial = np.array([1] * N, dtype=np.float32)
initial /= sum(initial)

transition = np.random.rand(N*N).reshape([N, N])
for i in xrange(0, len(transition)):
	transition[i,:] *= 1. / sum(transition[i,:])

test_transition = np.eye(N) + .006
for i in xrange(len(test_transition)):
	test_transition[i] /= sum(test_transition[i])
	# np.random.shuffle(test_transition[i])





# POSTERIOR DECODING TESTING
# T = 500
# pos = meth_data['pos'][0:L]
# model = hmm.Hmm(states, initial, test_transition)
# path, obs, cov = model.generateData(meth_data['cov'],
# 	cov_type="real", seq_length = T, fix_cov_val = 3)
# best_path, acc = model.decode(pos, obs, cov, 
#  	method='posterior',
#  	real_path = path,
#  	graph=True,
#  	ci_alpha = .90)

# Whole run on real dataset
T = 1000
pos = meth_data['pos'][0:L]
obs = meth_data['count'][0:L]
cov = meth_data['cov'][0:L]
model = hmm.Hmm(states, initial, transition)
Dkl = model.train(pos, obs, cov, maxiter = 30)
best_path = model.decode(pos, obs, cov, 
	method='posterior',
	real_path = None,
	graph=True,
	ci_alpha = .90)
