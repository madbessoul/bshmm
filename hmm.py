import numpy as np
import sys
import time
from scipy import stats
import scikits.bootstrap as bs
import ipdb

# HELPER FUNCTIONS


def stoch_check(matrix):
# Function to check if a matrix is row-stochastic (rows summing to 1)

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
    # in fact, the value and their machine representation are not always
    # the same. Instead, we make sur the difference is "close enough" to 0
    if (ones - ssum < 10e-8).all():
        return True
    else:
        return False


def stationaryDist(matrix):
# Compute the stationary distribution of a Markov Model based on the
# transition matrix. We use eigen decomposition for obvious performance
# gains.

    S, U = np.linalg.eig(matrix.T)
    mu = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    mu = mu / np.sum(mu)
    return mu


def simPoissonCoverage(lam, length):
    # Simulation of a poisson coverage of size "length"

    cov = np.random.poisson(lam, size=length)
    for i in xrange(length):
        if cov[i] <= 0:
            while cov[i] <= 0:
                cov[i] = np.random.poisson(lam, size=1)
    return cov


def simFloatWindowCoverage(real_cov, seq_length, win_length=100):
    # Simulation of flaoating window coverage based on real data
    # The purpose is to replicate local coverage variabilit as much as possible
    # We use fixed size floating window (default = 100) around each position,
    # and we resample the coverage at this position using the distribution
    # inside the window.

    sampled_cov = np.zeros([seq_length], int)
    for i in xrange(seq_length):

        # Fetch the window content to be used a sampling distribution
        if i < win_length / 2:
            window_content = real_cov[:i + win_length / 2]
        elif i > seq_length - win_length / 2:
            window_content = real_cov[i - win_length / 2:]
        else:
            window_content = real_cov[i - win_length / 2:i + win_length / 2]

        # Construct the distribution of the windowed coverage
        dist, bins = np.histogram(window_content,
            bins=np.linspace(0, len(window_content) + 1,
            max(window_content) + 1)
        )
        dist = np.array(dist, float)
        dist /= sum(dist)

        # Sample a value for the coverage at i from the dist distribution
        # whithout letting it be equal or in inf than 0
        while sampled_cov[i] <= 0:
            sampled_cov[i] = np.random.choice(range(0, max(window_content)),
                size=1, p=dist)

    return sampled_cov


    def conf_interval(array, alpha):
    
        # Return the confidence intervals
        best_index = np.argmax(array)
        best_val = max(array)

        try:
            ci_max = np.max(np.where(array[best_index:] \
                >= best_val - alpha))
        except:
            ci_max = 0

        try:
            ci_min = np.min(np.where(array[:best_index] \
                >= best_val - alpha))
        except:
            ci_min = 0

        return ci_max, ci_min


class Hmm:
    def __init__(self, states, initial, transition):
    # Class constructor with HMM arting parameters
    # N States
    # M Obsevations
    # Initial probability distribution with N values
    # N x N transition matrix
    # N x M emission matrix
    
        self.states = states

        if stoch_check(initial):
            self.initial = initial
        else:
            print "Initial matrix must be stochastic (rows must sum to 1)"


        if stoch_check(transition):
            self.transition = transition
        else: 
            print "Transition matrix must be stochastic (rows must sum to 1)"

        self.N = len(states)


    def _kl_divergence(self, P, Q):
    # Compute the Kullback-Leibler divergence for two distributions P and Q
        
        # Calculate the stationary distribution mu using eigen decomposition
        mu_P = stationaryDist(P)
        Dkl = 0
        N, M = P.shape
        for i in xrange(N):
            for j in xrange(M):
                fact = P[i,j] * mu_P[i]
                Dkl += fact*(np.log2(P[i,j]/Q[i,j]))
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
        
    #   new_emission = np.array(new_emission)
    #   if stoch_check(new_emission):
    #       self.emission = new_emission
    #   else:
    #       print "New emission matrix must be stochastic (rows must sum to 1)"


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
    #             = Binomial(n = c_counts, p = methylation, k = coverage)

    
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
            beta[:,t] *= 1. / sum(beta[:,t])

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

        return q_star

    def train(self, train_set, maxiter=30, threshold=10e-10, graph=False):
        '''
        Train the model using the Baum-Welch algorithm and update the model's
        current parameters based on a set training observation, a max number of 
        iteration and a threshold for the likelihood difference between new and
        current model
            - maxiter :     maximum number of EM iterations. Default = 100
            - threshold : minimum value for log-likehood difference between two
                        iterations. If lower, the EM stops. Default = 10e-10
            - graph :   plot the LL evolution at the end of the estimation. 
                        default = True  
        '''

        print " --- Parameter reestimation ---"


        initial_transition = self.transition
        start  = time.time()
        LLs = []
        for i in xrange(maxiter):
            # Print current EM iteration
            sys.stdout.write("\r   EM Iteration : %i/%i" % (i+1, maxiter))
            sys.stdout.flush()

            # Run Baum-Welch and store the log-likelihood prob
            LL = self._baumWelch(train_set)
            LLs.append(LL)

            # # Stop if the LL plateau is reached
            # if (i > 2):
            #   if (LLs[-1] - LLs[-2] < threshold):
            #       print '\nOops, log-likelihood plateau, training stopped'
            #       break
        stop = time.time()
        print "\t... done in %.1f secs" % (stop - start)

        # Plot LL evolution for each iteration
        # if graph is True:
        #   from pylab import plot, title, show
        #   plot(LLs, '+')
        #   title('Log-likelihood evolution during training')
        #   show()

        # Compute the KL divergence between initial transition and estimated
        # transition

        Dkl = self._kl_divergence(initial_transition, self.transition)
        return Dkl
 
    def _viterbiDecode(self, obs, coverage):
        '''
        Run the Viterbi algorithm to find the most probable state path for a 
        given set of observations
        '''

        print " --- Viterbi decoding ---"
        start_decode = time.time()
        T = len(pred_set)

        counts = obs
        cov = coverage
        # The notations (ie the variable names) correspond the those used in 
        # the famous Rabiner article (A Tutorial on Hidden Markov Models and 
        # Selected Applications in Speech Recognition, Feb. 1989)
        
        # Initialization
        
        e0 = self._binomialEmission(counts, cov, 0)

        delta = np.zeros([self.N, T], float)
        delta[:,0] = np.log(self.initial) + np.log(e0)

        psi = np.zeros([self.N, T], int)
        # Recursion
        for t in xrange(1,T):

            e_t = self._binomialEmission(counts, cov, t)
            tmp = delta[:, t-1] + np.log(self.transition)
            delta[:, t] = tmp.max(1) + np.log(e_t)
            psi[:, t] = tmp.argmax(1)

        # Backtracking
        q_star = [np.argmax(delta[:, T-1])]
        for t in reversed(xrange(T-1)):
            q_star.insert(0, psi[q_star[0], t+1])

        end_decode = time.time()
        print "\t... done in %.1f seconds" % (end_decode - start_decode)

        best_path = [self.states[i] for i in best_path_index]
        return best_path

    def _posteriorDecode(self, obs, coverage):
        '''
        Find the most probable hidden state sequence using the posterior
        probability and the forward-backward 
        '''
        
        counts = obs
        cov = coverage      

        T = len(counts)

        # TIMESTAMP
        print ' --- Posterior decoding ---'
        post_decode_start = time.time()

        log_obs, alpha, scale = self._forward(counts, cov)
        beta = self._backward(counts, cov, scale)

        Pkx = np.zeros([self.N, T])
        ci = np.zeros([2, T])
        Px = sum(alpha[:,T-1])
        
        # Compute posterior probabilities alpha(t) * beta(t) * 1/P(x)
        # for each position
        ci_alpha = .45
        for t in xrange(T-1):
            Pkx[:,t] = (1. / Px) * alpha[:,t] * beta[:,t]

            # Normalize posterior probabilities
            Pkx[:,t] *= 1. / sum(Pkx[:,t])

            # Confidence interval, alpha = ci_alpha
            ci[0,t], ci[1,t] = conf_interval(Pkx[:,t], ci_alpha)


        # Find the argmax of the posterior probability and return
        # the best methylation level sequence taken from the 
        # arrays of states. Same for CIs

        best_path = [self.states[i] for i in np.argmax(Pkx, 0)]
        ci[0,:] = [self.states[i] for i in ci[0,:]]
        ci[1,:] = [self.states[i] for i in ci[1,:]]


        # TIMESTAMP
        post_decode_stop = time.time()
        print '\t...done in %.1f secs' % (post_decode_stop - post_decode_start)

        return best_path, ci

    def decode(self, obs, coverage, method="posterior"):
        '''
        Find the best methylation state sequence. Arguments :
        method:
                "posterior"     use posterior decoding and fwd-bwd algorithms to infer
                                to infer methylation profile. Also returns confidence
                                intervals for each position.
                "viterbi"       use the Viterbi algorithm to find the most probable path
                                over the sequence
        '''

        # Call the selected algorithm for decoding
        if method=="posterior":
            best_path, ci = _posteriorDecode(obs, cov)
            return best_path, ci

        elif method=="viterbi":
            best_path = _viterbiDecode(obs, cov)
            return best_path


    def generateData(self, coverage, 
        cov_type="real", 
        fix_cov_val = 30, 
        seq_length=1000,
        window=100,
        lam = 15):

        '''
        Use the HMM as a generative model to simulate data using the model
        given parameters. We generate 3 types of data : state sequence, obs.
        sequence (C counts) and coverage data, with one of the following method,
        set by the cov_type argument:

            "real"          Uses real coverage data, from sampileup file for example.
                            In this case, we DO NOT simulate coverage data.
            "fixed"         Fixed coverage, use fix_cov_val to set the coverage value
            "window"       Simulated coverage using a floating window, based on
                            on real coverage data. Use the "window" argument to 
                            set the window size
            "poisson"       Simulation coverage ~ a poisson distribution
                            use the "lam" argument to set the parameter of
                            the poisson distribution
        '''

        counts = np.zeros([seq_length])
        gen_path = np.zeros([seq_length])

        # Sample initial state and initial observation
        gen_path[0] = np.random.choice(self.states, p=self.initial)
        counts[0] = np.random.binomial(coverage[0], gen_path[0], size=1)

        print ' --- Simulating model based data --- '
        simu_start = time.time()

        # Simulating coverage data
        if cov_type == "real":
            print "      - using real coverage data"
            cov = coverage

        elif cov_type == "fixed":
            print "      - using fixed (%d)coverage data" % fix_cov_val
            cov = np.ones([seq_length]) * fix_cov_val

        elif cov_type == "window":
            print "      - floating window coverage simulation \(size = %d)" \
                % window
            cov = simFloatWindowCoverage(coverage, seq_length, window)

        elif cov_type == "poisson":
            print "      - Poisson coverage simulation (lam = %d)" % lam
            cov = simPoissonCoverage(lam, seq_length)


        simu_end = time.time()
        print "\t...done in %.1f secs" % (simu_end - simu_start)


        # Sample the state paths and the corresponding observations
        for i in xrange(1, seq_length):
        
            curr_trans_dist = np.array(self.transition[
            np.where(
                self.states == gen_path[i - 1]
                )[0][0]
            , :], dtype=np.float32)

            # Sample the current state from the adequate transition matrix
            # line distribution
            gen_path[i] = np.random.choice(self.states, p=curr_trans_dist)

            # Sample the observation using a binomial distribution
            try:
                counts[i] = np.random.binomial(cov[i], gen_path[i], 1)
            except:
                ipdb.set_trace()


        return gen_path, counts, cov