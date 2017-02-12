# WORKSHOP-STUFF


Trainer.py creates a Gaussian Mixture Model w/ MFCC's using an Audio file, for both male and female classes. 

TesterOfModel.py contains 60 pieces of test data and tests the model for ability to discriminate between male and female voices. Model currently has a 93.3% success rate using this test data as an accuracy measure. 


FEMALE TRAINING AUDIO DATA: https://www.dropbox.com/s/bhc9vnixuz81tx4/Female1.wav?dl=0

MALE TRAINING AUDIO DATA: https://www.dropbox.com/s/dpawvo4zspgupoc/Male1.wav?dl=0

All of the test audio file is given through links (contained on the TesterOfModel.py file) which are parsed through a web-scrapper, audio contents extracted, saved to a .wav file on the desk, resaved for the next file, and so on. 
