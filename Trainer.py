from bs4 import BeautifulSoup
from urllib2 import urlopen
import urllib
import urllib2
import requests 
from sklearn import mixture
from python_speech_features import mfcc
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
from sklearn import preprocessing




BOY_VOICE = ["http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=150",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=152",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=153",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=155",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=160",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=161",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=162",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=163",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=416",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=444",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=446",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=465",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=480",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=489",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=497",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=507",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=509",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=511",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=517",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=518",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=522",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=526",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=527",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=528",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=535",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=538",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=547",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=551"]

GIRL_VOICE = ["http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=445",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=469",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=487",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=490",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=504",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=510",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=523",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=540",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=542",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=546",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=548",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=550",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=556",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=573",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=574",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=597",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=605",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=606",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=636",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=639",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=667",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=668",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=674",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=679",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=684",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=739",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=747",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=748",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=749",
"http://accent.gmu.edu/searchsaa.php?function=detail&speakerid=764"]






for t in BOY_VOICE:
	page = urllib.urlopen(t, 'lxml')
	html = page.read()
	soup = BeautifulSoup(html)
	x = soup.audio.source
	y = x.get('src')
	urllib.urlretrieve (y, "/Users/abhishaikemahajan/Documents/VOICE.mp3")
	AudioSegment.from_mp3("/Users/abhishaikemahajan/Documents/VOICE.mp3").export("/Users/abhishaikemahajan/Documents/VOICE.wav", format="wav")
	(RATE_INPUT,SIG_INPUT) = wav.read("/Users/abhishaikemahajan/Documents/VOICE.wav")  
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
	

for t in GIRL_VOICE:
	page = urllib.urlopen(t, 'lxml')
	html = page.read()
	soup = BeautifulSoup(html)
	x = soup.audio.source
	y = x.get('src')
	urllib.urlretrieve (y, "/Users/abhishaikemahajan/Documents/VOICE.mp3")
	AudioSegment.from_mp3("/Users/abhishaikemahajan/Documents/VOICE.mp3").export("/Users/abhishaikemahajan/Documents/VOICE.wav", format="wav")
	(RATE_INPUT,SIG_INPUT) = wav.read("/Users/abhishaikemahajan/Documents/VOICE.wav")  
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
