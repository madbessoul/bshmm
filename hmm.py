import numpy as np
import sys, time
from scipy import stats
import ipdb

# HELPER FUNCTIONS
# 
def stoch_check(matrix):
# Function to check if a matrix is row-stochastic (rows summing to 1)
# 
	
	if matrix.ndim == 1:
		ssum = np.sum(matrix)
		M = 1
	else:
		# Sum of rows
		ssum = np.sum(matrix, 1)
		M = np.size(matrix, 0)
	# Vector of [1, 1, ..., 1], M times, M being the number of rows	
	ones = np.array([1] * M)


	# Since we're dealing with float point values, 
	# it's not safe to test for equality,
	# in fact, the value and their machine representation are not always the same
	# Instead, we make sur the difference is "close enough" to 0
	if (ones - ssum < 10e-8).all():
		return True
	else:
		return False


class Hmm:
	def __init__(self, states, observations, initial, transition):
	# Class constructor with HMM arting parameters
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


		if stoch_check(transition):
			self.transition = transition
		else: 
			print "Transition matrix must be stochastic (rows must sum to 1)"

		self.N = len(states)
		self.M = len(observations)

	def kl_divergence(self, P, Q):
	# Compute the Kullback-Leibler divergence for two distributions P and Q
		
		# Calculate the stationary distribution mu using eigen decomposition
		S,U = np.linalg.eig(P.T)
		mu_P = np.array(U[:,np.where(np.abs(S-1.) < 1e-8)[0][0]].flat)
		mu_P = mu_P / np.sum(mu_P)
		
		Dkl = 0
		N, M = P.shape
		for i in xrange(N):
			for j in xrange(M):
				# ipdb.set_trace()
				# Dkl += P[i, j] * (np.log(P[i,j] / np.log(Q[i,j])))
				fact = P[i,j] * mu_P[i]
				Dkl += fact*(np.log2(P[i,j]) / np.log2(Q[i,j]))
		return Dkl

	def updateTransition(self, new_transition):
	# Set a new transition matrix for the model

		new_transition = np.array(new_transition)
		if stoch_check(new_transition):
			self.transition = new_transition
		else:
			print "Transition matrix mut be stochastic (row must sum to 1)"


	# def updateEmission(self, new_emission):
	# # Set a new emission matrix for the model
		
	# 	new_emission = np.array(new_emission)
	# 	if stoch_check(new_emission):
	# 		self.emission = new_emission
	# 	else:
	# 		print "New emission matrix must be stochastic (rows must sum to 1)"


	def updateInitial(self, new_initial):
	# Set a new initial distribution for the model
	# Also used by the Baum Welch algorithm to estimate optimal param.
	
		new_initial = np.array(new_initial)
		if stoch_check(new_initial):
			self.initial = new_initial
		else:
			print "New initial vector must be stochastic (row must sum to 1)"


	def _binomialEmission(self, obs, coverage, t):
	# Compute the binomial emission matrix at a time t
	# The Binomial emission matrix is defined as follows:
	# B(obs, state) = P(obs_i|state_i)
	# 			  = Binomial(n = c_counts, p = methylation, k = coverage)

	
		e = (stats.distributions.binom.pmf(obs[t], coverage[t], self.states))
		e *= 1. / sum(e)

		# The return vector must be stochastic, so we divide by the sum of its
		# elements 
		return e


	def _forward(self, obs, coverage):
	# Compute P(observation sequence | state sequence)
	# the forward algorithm and calculate the alpha variable as well as 
	# the observed sequences probability
	# Returns the alpha variable, the log probability of the observed
	# sequence as well as as the scaling factor (used in the backward alg.)

		# Variable initializaiton 
		T = len(obs)
		scale = np.zeros([T], float)
		alpha = np.zeros([self.N, T], float)

		### Initialisation step
		# c is the normalization value
		# Since we can't use log values to avoid underflow during computation, 
		# we normalize the probabilities using a scaling value while keeping 
		# the recursion coherent 

		e0 = self._binomialEmission(obs, coverage, 0)
		alpha[:,0] = self.initial * e0
		
		scale[0] = 1. / np.sum(alpha[:,0])
		alpha[:,0] *= scale[0]
		### Induction step (recursion)
		for t in xrange(1, T):
	
			e = self._binomialEmission(obs, coverage, t)
			alpha[:,t] = np.dot(alpha[:,t-1], self.transition) * e


			scale[t] = 1. / np.sum(alpha[:,t])
			alpha[:,t] *= scale[t]

		obs_log_prob = - np.sum(np.log(scale))


		return obs_log_prob, alpha, scale
		


	def _backward(self, obs, coverage, scale):
		'''
		Run the backward algorithm and calculate the beta variable
		'''

		T = len(obs) 
		beta = np.zeros([self.N, T])

		# Initialization
		beta[:,T-1] = 1.0
		beta[:,T-1] *= scale[T-1]
		
		# Induction 
		# Reverse iteration from T-1 to 0
		for t in reversed(xrange(T-1)):

			e = self._binomialEmission(obs, coverage, t+1)
			beta[:,t] = np.dot(self.transition, (e * beta[:,t+1]))

			# Scaling using the forward algorithm scaling factors
			beta[:,t] *= scale[t]

		return beta


	def _ksi(self, obs, coverage, alpha, beta):
	# Calculate ksis. 
	# Ksi is a N x N x T-1 matrix

		T = len(obs)
		ksi = np.zeros([self.N, self.N, T-1])

		for t in xrange(T-1):
			for i in xrange(self.N):
				e = self._binomialEmission(obs, coverage, t+1)
				ksi[i, :, t] = alpha[i, t] * \
					self.transition[i, :] * \
					e * \
					beta[:, t+1]

		return ksi


	def _gamma(self, alpha, beta):
	# Compute gammas
		
		gamma = alpha * beta 
		gamma /= gamma.sum(0)
		return gamma


	def _baumWelch(self, train_set):
	# Run the Baum-Welch algorithm to estimate new paramters based on 
	# a set of obsercations
	
		counts = train_set['count']
		coverage = train_set['cov']
		T = len(counts)
		
		log_obs, alpha, scale = self._forward(counts, coverage)
		beta = self._backward(counts, coverage, scale)
		ksi = self._ksi(counts, coverage, alpha, beta)
		gamma = self._gamma(alpha, beta)

		# Expectation of being in state i
		expect_si_all = np.zeros([self.N], float)

		# Expectation of being in state i until T-1
		expect_si_all_TM1 = np.zeros([self.N], float)

		# Exctation of jumping from state i to state j
  		expect_si_sj_all = np.zeros([self.N, self.N], float)

  		# Exctation of jumping from state i to state j until T-1
  		expect_si_sj_all_TM1 = np.zeros([self.N, self.N], float)


  		expect_si_all += gamma.sum(1)
  		expect_si_all_TM1 += gamma[:, :T-1].sum(1)
		expect_si_sj_all += ksi.sum(2)
		expect_si_sj_all_TM1 += ksi[:, :, :T-1].sum(2) 

  		### Update transition matrix
  		new_transition = np.zeros([self.N, self.N])
  		for i in xrange(self.N):
  			new_transition[i,:] = expect_si_sj_all_TM1[i,:] / \
  								expect_si_all_TM1[i]
  			new_transition[i,:] /= sum(new_transition[i,:])
  			# ipdb.set_trace()

  		self.updateTransition(new_transition)

  		# Return log likelihood
		return log_obs

	def _viterbi(self, pred_set):
	# Find the most probable hidden state sequence using the Viterbi algorithm
		T = len(pred_set)

		counts = pred_set['count']
		coverage = np.ones([T]) * 100

		# The notations (ie the variable names) correspond the those used in 
		# the famous Rabiner article (A Tutorial on Hidden Markov Models and 
		# Selected Applications in Speech Recognition, Feb. 1989)
		
		# Initialization
		
		e0 = self._binomialEmission(counts, coverage, 0)

		delta = np.zeros([self.N, T], float)
		delta[:,0] = np.log(self.initial) + np.log(e0)

		psi = np.zeros([self.N, T], int)
		# Recursion
		for t in xrange(1,T):

			e_t = self._binomialEmission(counts, coverage, t)
			tmp = delta[:, t-1] + np.log(self.transition)
			delta[:, t] = tmp.max(1) + np.log(e_t)
			psi[:, t] = tmp.argmax(1)

		# Backtracking
		q_star = [np.argmax(delta[:, T-1])]
		for t in reversed(xrange(T-1)):
			q_star.insert(0, psi[q_star[0], t+1])

		return (q_star, delta, psi)

	def train(self, train_set, maxiter=30, threshold=10e-10, graph=True):
		'''
		Train the model using the Baum-Welch algorithm and update the model's
		current parameters based on a set training observation, a max number of 
		iteration and a threshold for the likelihood difference between new and
		current model
			- maxiter : 	maximum number of EM iterations. Default = 100
			- threshold : minimum value for log-likehood difference between two
						iterations. If lower, the EM stops. Default = 10e-10
			- graph : 	plot the LL evolution at the end of the estimation. 
						default = True  
		'''

		print " --- Parameter reestimation : EM ---"
		start  = time.time()
		LLs = []
		for i in xrange(maxiter):
			# Print current EM iteration
			sys.stdout.write("\r   EM Iteration : %i/%i" % (i+1, maxiter))
			sys.stdout.flush()

			# Run Baum-Welch and store the log-likelihood prob
			LL = self._baumWelch(train_set)
			LLs.append(LL)

			# Stop if the LL plateau is reached
			if (i > 2):
				if (LLs[-1] - LLs[-2] < threshold):
					print '\nOops, log-likelihood plateau, training stopped'
					break
		stop = time.time()
		print "\t... done in %d secs" % (stop - start)

		# Plot LL evolution for each iteration
		if graph is True:
			from pylab import plot, title, show
			plot(LLs, '+')
			title('Log-likelihood evolution during training')
			show()
		return LL[-1], LL
 
	def viterbiDecode(self, obs):
		'''
		Run the Viterbi algorithm to find the most probable state path for a 
		given set of observations
		'''

		print " --- Decoding using Viterbi algorithm ---"
		start_decode = time.time()
		best_path, delta, psi = self._viterbi(obs)
		end_decode = time.time()
		print "\t... done in %d seconds" % (end_decode - start_decode)

		return (best_path, delta, psi)

	def posteriorDecode(self, pred_set):
	# Find the most probable hidden state sequence using the posterior probability
	# and the forward-backward algorithm
		
		counts = pred_set['count']
		coverage = pred_set['cov']
		T = len(counts)
		
		log_obs, alpha, scale = self._forward(counts, coverage)
		beta = self._backward(counts, coverage, scale)

		Pkx = np.zeros([self.N, T])
		Px = sum(alpha[:,T-1])
		
		for t in xrange(T-1):
			Pkx[:,t] = (1. / Px) * alpha[:,t] * beta[:,t]
		best_path = np.argmax(Pkx, 0)
		return best_path


	def generateData(self, coverage, seq_length=1000):
		'''
		Use the HMM as a generative model to simulate data using the model
		given parameters
		'''

		counts = np.zeros([seq_length])
		gen_path = np.zeros([seq_length])

		# Sample initial state and initial observation
		gen_path[0] = np.random.choice(self.states, p=self.initial)
		counts[0] = np.random.binomial(coverage[0], gen_path[0], size=1)

		print ' --- Simulating model based data --- '
		simu_start = time.time()
		# Sample the state paths and the corresponding observations
		for i in xrange(1, seq_length):
		
			curr_trans_dist = np.array(self.transition[
			np.where(
				self.states == gen_path[i-1]
				)[0][0]
			, :], dtype=np.float32)

			# Sample the current state from the adequate transition matrix
			# line distribution
			gen_path[i] = np.random.choice(self.states, p=curr_trans_dist)

			# Sample the observation using a binomial distribution
			counts[i] = np.random.binomial(coverage[i], gen_path[i], 1)
		simu_end = time.time()
		print "\t... done in %d secs" % (simu_end - simu_start)
		return gen_path, counts