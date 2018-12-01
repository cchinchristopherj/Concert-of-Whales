import numpy as np
import glob
import aifc
import scipy.signal as sp
import scipy
import math
import codecs
import json
from math import sqrt
import tensorflowjs as tfjs
from scipy.signal import butter, lfilter, freqz, filtfilt
import array
import wave
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import scipy.signal as sp
import scipy
import copy
from scipy.io import wavfile
import librosa
import librosa.display
import IPython.display as ipd
import pylab as pl
from matplotlib import mlab
import tensorflow as tf
import keras.optimizers
from keras.optimizers import Adam
import keras.regularizers
from keras import initializers
from keras.callbacks import Callback
from keras import layers
from keras.layers import Input, Dense, TimeDistributed, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda, GRU, Reshape, Embedding, Dropout, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine import InputSpec
K.set_image_data_format('channels_last')

# Code from: https://github.com/nmkridler/moby/blob/master/fileio.py
def ReadAIFF(file):
    ''' ReadAIFF Method
            Read AIFF and convert to numpy array
            
            Args: 
                file: string file to read 
            Returns:
                numpy array containing whale audio clip      
                
    '''
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.frombuffer(strSig,np.short).byteswap()

# Based off H1Sample Function from: https://github.com/nmkridler/moby/blob/master/fileio.py
def SpecGram(file,params=None):
    ''' SpecGram Method 
            Convert audio file to spectrogram for CNN and pre-process input for
            input shape uniformity 
            
            Args:
                file: string file to read 
                params: dictionary containing spectrogram parameters  
            Returns: 
                Pre-Processed Spectrogram matrix and frequency/time bins as 1-D arrays
                
    '''
    s = ReadAIFF(file)
    # Convert to spectrogram 
    P,freqs,bins = mlab.specgram(s,**params)
    m,n = P.shape
    # Ensure all image inputs to the CNN are the same size. If the number of time bins 
    # is less than 59, pad with zeros 
    if n < 59:
        Q = np.zeros((m,59))
        Q[:,:n] = P
    else:
        Q = P
    return Q,freqs,bins

def extract_labels(file):
    ''' extract_labels Method 
            Since the dataset file names contain the labels (0 or 1) right before
            the extension, appropriately parse the string to obtain the label 
            
            Args:
                file: string file to read 
            Returns: 
                int label of the file (0 or 1) 
                
    '''
    name,extension = os.path.splitext(file)
    label = name[-1]
    return int(label)

def minmaxscaling(X,minmaxscaler,flag=1):
    ''' minmaxscaling Method 
            Scales the input to the desired range using sklearn's MinMaxScaler()
            
            Args:
                X: Dataset 
                minmaxscaler: Instance of sklearn's MinMaxScaler() with pre-defined feature range
                flag: 1 indicates MinMaxScaler() should be fit to data then transform it, while
                      0 indicates MinMaxScaler() should solely transform the data
            Returns: 
                Scaled verion of X
                
    '''
    # Dimensions of X
    num_samples = X.shape[0]
    height = X.shape[1]
    width = X.shape[2]
    # Reshape X into a 2D array for MinMaxScaler()
    dataset = X.reshape((num_samples,height*width))
    dataset = np.swapaxes(dataset,0,1)
    # Flag value of 1 indicates MinMaxScaler() has already been fit
    if flag == 1:
        dataset = minmaxscaler.fit_transform(dataset)
    elif flag == 0:
        dataset = minmaxscaler.transform(dataset)
    # Reshape dataset into the original shape
    dataset = np.swapaxes(dataset,0,1)
    dataset = dataset.reshape((num_samples,height,width))
    X = dataset
    return X

# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    ''' reconstruct_signal_griffin_lim Method
            Reconstruct an audio signal from a magnitude spectrogram.
            Given a magnitude spectrogram as input, reconstruct
            the audio signal and return it using the Griffin-Lim algorithm from the paper:
            "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
            in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
            
            Args:
                magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
                    and the columns correspond to frequency bins.
                fft_size (int): The FFT size, which should be a power of 2.
                hopsamp (int): The hope size in samples.
                iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
                    is sufficient.
            Returns:
                The reconstructed time domain signal as a 1-dim Numpy array.
                
    '''
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        diff = sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct
    
# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def stft_for_reconstruction(x, fft_size, hopsamp):
    ''' stft_for_reconstruction Method
            Compute and return the STFT of the supplied time domain signal x.
            
            Args:
                x (1-dim Numpy array): A time domain signal.
                fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
                hopsamp (int):
            Returns:
                The STFT. The rows are the time slices and columns are the frequency bins.
                
    '''
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size]) for i in range(0, len(x)-fft_size, hopsamp)])

# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def istft_for_reconstruction(X, fft_size, hopsamp):
    ''' istft_for_reconstruction Method
            Invert a STFT into a time domain signal.
            
            Args:
                X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
                fft_size (int):
                hopsamp (int): The hop size, in samples.
            Returns:
                The inverse STFT.
                
    '''
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x

# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def save_audio_to_file(x, sample_rate, outfile='out.wav'):
    ''' save_audio_to_file Method
            Save a mono signal to a file.
            
            Args:
                x (1-dim Numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
                sample_rate (int): The sample rate of the signal, in Hz.
                outfile: Name of the file to save.
                
    '''
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x = x*32767.0
    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)
    f = wave.open(outfile, 'w')
    f.setparams((1, 2, sample_rate, 0, "NONE", "Uncompressed"))
    f.writeframes(data.tobytes())
    f.close()
    
def fix_batch(X_train,num_to_add):
    ''' fix_batch Method
            Select "num_to_add" random samples from the training set and add them to the end of the training
            set to ensure that the total number of samples in the training set is a multiple of the batch size
            
            Args:
                X_train: Training set
                num_to_add: Number of random samples from the training set to append to the end
            Returns:
                Training set with "num_to_add" newly-appended samples
                
    '''
    for ii in range(num_to_add):
        ind = np.random.randint(X_train.shape[0])
        X_train = np.append(X_train,X_train[ind,:,:][np.newaxis,...],axis=0)
    return X_train
    
# The neural network model outputs raw spectrogram frames, which can subsequently be converted to audio waveforms 
# using the Griffin-Lim algorithm
# The architecture is based on the Tacotron model from the original paper, which implements a CBHG module
# (1-D convolution bank + highway network + Bidirectional GRU) capable of extracting excellent representations from 
# sequences by convolving the sequence first with a bank of 1-D convolutional filters to extract local information, 
# passing it through a highway network to extract higher-level features, and finally passing the sequence through a 
# Bidirectional GRU to learn long-term dependencies in the forward and backward directions. In the Tacotron model, 
# an encoder uses this CBHG module to extract a sequential representation of input text, which the attention-based
# decoder uses to create a sequence of spectrogram frames that can be used to synthesize the correspnoding waveform.
# For simplicity, the decoder targets are 80-band mel spectrograms (a compressed representation that can be used by 
# a post-processing-net later on in the model to synthesize raw spectrograms). This post-processing-net is once again
# composed of a CBHG module, which learns to predict spectral magnitudes on a linear frequency scale due to the use
# of the Griffin-Lim algorithm to create waveforms.
def create_model(batch_input_shape,flag=1,rnn_dim=RNN_DIM,mel_dim=MEL_DIM,linear_dim=LINEAR_DIM,bank_dim=BANK_DIM):
    ''' create_model Method
            Create neural network model
            
            Args:
                batch_input_shape: Input shape including batch axis
                flag: 1 indicates training, while 0 indicates prediction
                rnn_dim: Dimension of GRUs
                mel_dim: Number of mel bands
                linear_dim: Number of bins in linear frequency scale
                bank_dim: Number of convolutional filters per layer
            Returns:
                Neural network model
                
    '''
    input_shape = (batch_input_shape[1],batch_input_shape[2])
    X_input = Input(batch_shape=batch_input_shape,name='input')
    prenet1 = Dense(64,activation='relu',name='prenet1')
    X = TimeDistributed(prenet1,name='td_prenet1')(X_input)
    X = Dropout(0.5,name='dropout1')(X)
    prenet2 = Dense(64,activation='relu',name='prenet2')
    X = TimeDistributed(prenet2,name='td_prenet2')(X)
    X = Dropout(0.5,name='dropout2')(X)
    if flag == 1:
        rnn1 = Bidirectional(GRU(rnn_dim,name='rnn1',return_sequences=True))
    elif flag == 0:
        rnn1 = Bidirectional(GRU(rnn_dim,name='rnn1',return_sequences=True,stateful=True))
    X = rnn1(X)
    adapter1 = Dense(mel_dim,use_bias=False,name='adapter1')
    X_output1 = TimeDistributed(adapter1,name='td_adapter1')(X)
    X_bank1 = Conv1D(bank_dim,3,padding='same',activation='relu',name='bank1')(X_output1)
    X_bn1 = BatchNormalization(name='bn1')(X_bank1)
    X_bank2 = Conv1D(bank_dim,5,padding='same',activation='relu',name='bank2')(X_output1)
    X_bn2 = BatchNormalization(name='bn2')(X_bank2)
    X_bank3 = Conv1D(bank_dim,7,padding='same',activation='relu',name='bank3')(X_output1)
    X_bn3 = BatchNormalization(name='bn3')(X_bank3)
    X_bank = keras.layers.concatenate([X_bn1,X_bn2,X_bn3],name='concat')
    X = MaxPooling1D(strides=1,padding='same',name='maxpool1')(X_bank)
    adapter2 = Dense(bank_dim,use_bias=False,name='adapter2')
    X = TimeDistributed(adapter2,name='td_adapter2')(X)
    if flag == 1:
        rnn2 = Bidirectional(GRU(rnn_dim,name='rnn2',return_sequences=True))
    elif flag == 0:
        rnn2 = Bidirectional(GRU(rnn_dim,name='rnn2',return_sequences=True,stateful=True))
    X = rnn2(X)
    adapter3 = Dense(linear_dim,use_bias=False,name='adapter3')
    X_output2 = TimeDistributed(adapter3,name='td_adapter3')(X)
    model = Model(inputs=X_input,outputs=[X_output1,X_output2])
    return model

def sample(model_pred,X_train):
    ''' sample Method
            Use the trained neural network to make a prediction for a new spectrogram
            from the state space (and corresponding audio waveform)
            
            Args:
                model_pred: The model used to make predictions
                X_train: Training set
            Returns:
                samples: 2D array of the spectrogram generated by the neural network
                
    '''
    # Number of time steps in a spectrogram of the training set
    ts = X_train.shape[1]
    # Instantiate an array of zeros with the shape of a spectrogram from the training set
    samples = np.zeros((1,ts,X_train.shape[2]))
    # The neural network is tasked with predicting a new spectrogram (frame-by-frame) based
    # on information it learned from spectrograms of the training set 
    # Since the model makes predictions for every subsequent frame in a sequence, it can
    # only make predictions from the second frame up to the last frame. Therefore, the 
    # first frame in the sequence of spectrogram frames must be known (or assumed) a priori.
    # In this case, the first spectrogram frame from a random spectrogram in the training set
    # will be used as the first frame in the new sequence.
    ind = np.random.randint(X_train.shape[0])
    samples[:,0] = X_train[ind][0]
    # The model will take every frame starting from the first as input and make a prediction
    # for the subsequent frame. Save each spectrogram frame in the appropriate location in
    # "samples"
    for t in range(1, ts):
        sample_mel,sample_linear = model_pred.predict_on_batch([samples[:,t-1:t]])
        sample_linear = sample_linear.reshape((sample_linear.shape[1],sample_linear.shape[2]))
        samples[:, t] = sample_linear
    # Reshape "samples" into the shape expected by the reconstruct_signal_griffin_lim function
    # Then use the Griffin Lim algorithm to synthesize an audio waveform corresponding to the 
    # magnitude spectrogram
    samples = samples.reshape((samples.shape[1],samples.shape[2]))
    x_reconstruct = reconstruct_signal_griffin_lim(samples, fft_size=256, hopsamp=192, iterations=20)
    # Normalize the samples of the audio waveform
    x_reconstruct = x_reconstruct/max(abs(x_reconstruct))
    sample_rate=2000
    # Save the waveform as a .wav file
    save_audio_to_file(x_reconstruct,sample_rate,outfile='out.wav')
    return samples

# Spectrogram parameters 
params = {'NFFT':256,'Fs':2000,'noverlap':192}
# Load in the audio files from the training dataset
path = 'Documents/Bioacoustics_MachineLearning/train2'
filenames = glob.glob(path+'/*.aif')
# Extract labels for each file from the file names
Y_train = np.array([extract_labels(x) for x in filenames])
# Identify which samples of the dataset contain upcalls (labeled 1)
pos_indexes = np.where(Y_train==1)[0]
pos_filenames = list(np.array(filenames)[pos_indexes])
# For each audio file with upcall, extract the spectrograms 
X_train = np.array([SpecGram(x,params=params)[0] for x in pos_filenames])
# Option for instead extracting spectrograms with vertically-enhanced contrast
# X_train = np.array([extract_featuresV(x,params=params) for x in pos_filenames])

# Convert spectrograms to the 80-band mel scale
X_train_mel = np.array([librosa.feature.melspectrogram(S=D,sr=2000,n_mels=80) for D in X_train])

# Instance of MinMaxScaler() for linear spectrograms (default feature range of 0 to 1)
minmaxscaler_linear = MinMaxScaler()
# Instance of MinMaxScaler() for mel spectrograms (default feature range of 0 to 1)
minmaxscaler_mel = MinMaxScaler()
# Scale the datasets of linear spectrograms and mel spectrograms for input to the neural network
X_train = minmaxscaling(X_train,minmaxscaler_linear,flag=1)
X_train_mel = minmaxscaling(X_train_mel,minmaxscaler_mel,flag=1)

# Add 4 random samples from X_train and X_train_mel to the end of the respective datasets to ensure
# that the total number of samples in both datasets is a multiple of the batch size
num_to_add = 4
X_train = fix_batch(X_train,num_to_add)
X_train_mel = fix_batch(X_train_mel,num_to_add)

# The reconstruct_signal_griffin_lim algorithm requires the input spectrograms to have rows that 
# correspond to time slices and columns that correspond to frequency bins (the reverse of the shape
# output by the SpecGram function used to generate the training set). Therefore, swap the 1st and 
# 2nd axes of the X_train and X_train_mel datasets
X_train = np.swapaxes(X_train,1,2)
X_train_mel = np.swapaxes(X_train_mel,1,2)
# The input to the neural network will be X_train, the sequences of linear spectrogram frames, and the neural 
# network will be tasked with predicting the next frame for every input frame. In other words, given the first 
# frame, the model must predict the second. Given the second frame, the model must predict the third, and so on.
# For this reason, X_train must consist of spectrogram frames from the first to the second to last in each 
# sequence, since the model will make predictions for the second frame up to the last frame. 
# In addition, due to the architecture of the neural network, there will also be two sets of ground truth 
# annotations for, one for the mel spectrograms and the other for the linear spectrograms. These sets 
# of ground truth annotations are created by shifting X_train and X_train_mel one time step forward (i.e. 
# the ground truth annotations are for the second frame up to the last frame, corresponding to the predictions
# of the model).
Y_train_linear = np.copy(X_train)[:,1:,:]
Y_train_mel = np.copy(X_train_mel)[:,1:,:]
X_train = X_train[:,0:-1,:]

# Model parameters
# Batch size
n_batch=32
# Dimension of GRUs
RNN_DIM = 128
# Number of mel bands
MEL_DIM = 80
# Number of bins in linear frequency scale
LINEAR_DIM = 129
# Number of convolutional filters per layer
BANK_DIM = 128

# Specify the input shape, including the batch axis
batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2])
# Create the model and compile it. Note that due to the two outputs, two losses are computed.
# Give equal weight to both losses.
model_train = create_model(batch_input_shape)
model_train.compile(loss='mean_absolute_error',optimizer='adam',loss_weights=[0.5,0.5])

# Train the model
model_train.fit(X_train,[Y_train_mel,Y_train_linear],batch_size=32,epochs=10)
# After training save the weights of the model
file_name = 'taco.h5'
save_weights(file_name,model_train)

# Use the trained model to create a prediction for a new spectrogram from the state space
# A different iteration of the neural network will be created for prediction since the 
# batch size for prediction is equal to 1 (the model will be predicting on a frame-by-frame
# basis). The weights will be loaded into the corresponding layers from the trained neural 
# network
n_batch=1
batch_input_shape=(n_batch,1,X_train.shape[2])
model_pred = create_model(batch_input_shape,flag=0)
model_pred.compile(loss='mean_absolute_error',optimizer='adam',loss_weights=[0.5,0.5])
file_name = 'taco.h5'
load_weights(file_name,model_pred)
# Predict a new sequence of spectrogram frames and save the corresponding audio waveform
# as a .wav file
samples = sample(model_pred,X_train)



