from sklearn import mixture
from python_speech_features import mfcc
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from sklearn import preprocessing

##########################################################################################################################################


def delta(feat, N): #definition for delta function used in the MFCC feature extraction 
	NUMFRAMES = len(feat)
	feat = np.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
	denom = sum([2*i*i for i in range(1,N+1)])
	dfeat = []
	for j in range(NUMFRAMES):
		dfeat.append(np.sum([n*feat[N+j+n] for n in range(-1*N,N+1)], axis=0)/denom)
	return dfeat

##########################################################################################################################################


(RATE_BOY,SIG_BOY) = wav.read("/Users/abhishaikemahajan/Documents/VOICE/MALE-AMERICAN/Male1.wav")
MFCC_BOY = mfcc(SIG_BOY,RATE_BOY, numcep = 16)
DELTA1_BOY= delta(MFCC_BOY, 2)
DELTA2_BOY = delta(DELTA1_BOY,2)
BOY_FEATURES = pd.concat([pd.DataFrame(MFCC_BOY),pd.DataFrame(DELTA1_BOY),pd.DataFrame(DELTA2_BOY)],  axis=1) #36 dimensional dataframe of the MFCC, Delta1 (first derivative), and Delta1 (second derivative)
BOY_FEATURES = preprocessing.scale(BOY_FEATURES)

##########################################################################################################################################


(RATE_GIRL,SIG_GIRL) = wav.read("/Users/abhishaikemahajan/Documents/VOICE/FEMALE-AMERICAN/Female1.wav")
MFCC_GIRL = mfcc(SIG_GIRL,RATE_GIRL, numcep = 16)
DELTA1_GIRL= delta(MFCC_GIRL, 2)
DELTA2_GIRL = delta(DELTA1_GIRL,2)
GIRL_FEATURES = pd.concat([pd.DataFrame(MFCC_GIRL),pd.DataFrame(DELTA1_GIRL),pd.DataFrame(DELTA2_GIRL)],  axis=1)
GIRL_FEATURES = preprocessing.scale(GIRL_FEATURES)

##########################################################################################################################################


BOY_MODEL = mixture.GaussianMixture(n_components = 20, max_iter=1000, tol = .01, warm_start = True, covariance_type = 'diag')
BOY_MODEL.fit(BOY_FEATURES)

GIRL_MODEL = mixture.GaussianMixture(n_components = 20, max_iter=1000, tol = .01, warm_start = True, covariance_type = 'diag')
GIRL_MODEL.fit(GIRL_FEATURES)

##########################################################################################################################################

(RATE_INPUT,SIG_INPUT) = wav.read("/Users/abhishaikemahajan/Documents/VOICE/WORKSHOPTEST/SeshaTest.wav")  

MFCC_INPUT = mfcc(SIG_INPUT,RATE_INPUT, numcep = 16)
DELTA1_INPUT = delta(MFCC_INPUT, 2)
DELTA2_INPUT = delta(DELTA1_INPUT,2)

INPUT_FEATURE = pd.concat([pd.DataFrame(MFCC_INPUT),pd.DataFrame(DELTA1_INPUT), pd.DataFrame(DELTA2_INPUT)],  axis=1)
INPUT_FEATURE = preprocessing.scale(INPUT_FEATURE)

BOY_SCORE = (BOY_MODEL.score_samples(INPUT_FEATURE))
GIRL_SCORE = (GIRL_MODEL.score_samples(INPUT_FEATURE))

STATES = pd.DataFrame({0 : xrange(0, len(BOY_SCORE))})

for x in xrange(0,len(STATES)):
	if (BOY_SCORE[x] > GIRL_SCORE[x]):
		STATES[x:x+1] = 0;
	if (BOY_SCORE[x] < GIRL_SCORE[x]):
		STATES[x:x+1] = 1;

STATES = STATES.squeeze()
UnFormatedCounts = STATES.value_counts()

if (UnFormatedCounts[0] > UnFormatedCounts[1]):
	print "IT'S A BOY"

if (UnFormatedCounts[0] < UnFormatedCounts[1]):
	print "IT'S A GIRL"


###########################################
