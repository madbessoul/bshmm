import numpy as np

def stoch_check(matrix):
# Function to check if a matrix is row-stochastic (rows summing to 1)
# 
	
	if matrix.ndim == 1:
		ssum = np.sum(matrix)
		M = 1
	else:
		# Sum of rows
		ssum = np.sum(matrix, 1)
		M = np.size(matrix, 1)

	# Vector of [1, 1, ..., 1], M times, M being the number of rows	
	ones = np.array([1] * M)
	
	# Since we're dealing with float point values, it's not safe to test for equality,
	# in fact, the value and their machine representation are not always the same
	# Instead, we make sur the difference is "close enough" to 0
	if (ones - ssum < 10e-8).all():
		return True
	else:
		return False


class Hmm:
	def __init__(self, states, observations, initial, transition, emission):
	# Class constructor with HMM starting parameters
	# N States
	# M Obsevations
	# Initial probability distribution with N values
	# N x N transition matrix
	# N x M emission matrix
	
		self.states = states
		self.observations = observations

		if stoch_check(initial):
			self.initial = initial
		else:
			print "Initial matrix must be stochastic (rows must sum to 1)"

		if stoch_check(emission):
			self.emission = emission
		else:
			print "Emission matrix must be stochastic (rows must sum to 1)"

		if stoch_check(transition):
			self.transition = transition
		else: 
			print "Transition matrix must be stochastic (rows must sum to 1)"

		self.N = len(states)
		self.M = len(observations)


	def updateState(self, new_states):
		self.states = new_states

	def updateTransition(self, new_transition):
	# Set a new transition matrix for the model

		new_transition = np.array(new_transition)
		if stoch_check(new_transition):
			self.transition = new_transition
		else:
			print "Transition matrix mut be stochastic (row must sum to 1)"


	def updateEmission(self, new_emission):
	# Set a new emission matrix for the model
		
		new_emission = np.array(new_emission)
		if stoch_check(new_emission):
			self.emission = new_emission
		else:
			print "New emission matrix must be stochastic (rows must sum to 1)"


	def updateInitial(self, new_initial):
	# Set a new initial distribution for the model
	
		new_initial = np.array(new_initial)
		if stoch_check(new_initial):
			self.initial = new_initial
		else:
			print "New initial vector must be stochastic (row must sum to 1)"


	def _forward(self, states, observations, transition, emission):
	# Compute P(observation sequence | state sequence)
	# Run the forward algorithm and calculate the alpha variable

		pass

		# Initialisation step
		alpha = emission * observations

		# Induction step (recursion)


		# Termination
		
		# return alpha


	def _backward(self):
	# Run the backward algorithm and calculate the beta variable
	
		pass
		# return beta


	def _baumWelch(self, train_set):
	# Run the Baum-Welch algorithm to estimate new paramters based on 
	# a set of obsercations
	
		pass
		# alpha = _forward()
		# beta = _backward()
		# Compute xi and gamma
		# Estimate new parameters (transition, initial)


	def _viterbi(self, pred_set):
	# Find the most probable hidden state sequence using the Viterbi algorithm
	
		pass
		# return path, path_like


	def train(self, tain_set, maxiter, threshold):
	# Train the model using the Baum-Welch algorithm and update the model's
	# current parameters based on a set training observation, a max number of 
	# iteration and a threshold for the likelihood difference between new and
	# current model
	
		pass
		# _baumwelch(train_set)
		# updateTransition(train_transition)
		# updateInitial(train_initial)

