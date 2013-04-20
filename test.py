import hmm
import numpy as np


initial = 1./100 * np.array([25, 25, 25, 25], dtype=np.float32)

states = np.array([0., 5., .9, 1.], dtype=np.float32)

observations = np.array([0, 1, 2, 3], dtype = np.int32)

emission = 1./10 * np.array([
	[5, 2, 2, 1],
	[2, 5, 2, 1],
	[1, 2, 5, 2],
	[1, 2, 2, 5]
], dtype=np.float32)

transition = 1./10 * np.array([
	[7, 1, 1, 1],
	[1, 7, 1, 1],
	[1, 1, 7, 1],
	[1, 1, 1, 7]
], dtype=np.float32)

model = hmm.Hmm(states, observations, initial, transition, emission)