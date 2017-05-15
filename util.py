import numpy as np
import scipy
from scipy import io
import os
import sys

def infoAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    bitrate = audioObj.dtype    
    nsamples = len(audioObj)
    audioObj = None
    return nsamples, sampleRate, bitrate

def readAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    bitrate = audioObj.dtype    
    try:
        maxv = np.finfo(bitrate).max
    except:
        maxv = np.iinfo(bitrate).max
    return audioObj.astype('float')/maxv, sampleRate, bitrate

def writeAudioScipy(fileout,audio_out,sampleRate,bitrate="int16"):
    maxn = np.iinfo(bitrate).max  
    scipy.io.wavfile.write(filename=fileout, rate=sampleRate, data=(audio_out*maxn).astype(bitrate))

def saveTensor(self, t, name=''):
    """
    Saves a numpy array as a binary file
    """
    t.tofile(self.out_path.replace('.data',name+'.data'))
    #save shapes
    self.shape = t.shape
    self.save_shape(self.out_path.replace('.data',name+'.shape'),t.shape)

def loadTensor(self, name=''):
    """
    Loads a binary .data file
    """
    f_in = np.fromfile(self.out_path.replace('.data',name+'.data'))
    shape = self.get_shape(self.out_path.replace('.data',name+'.shape'))
    if self.shape == shape:
        f_in = f_in.reshape(shape)    
        return f_in
    else:
        print 'Shape of loaded array does not match with the original shape of the transform'

def save_shape(self,shape_file,shape):
    """
    Saves the shape of a numpy array
    """
    with open(shape_file, 'w') as fout:
        fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')

def get_shape(self,shape_file):
    """
    Reads a .shape file
    """
    with open(shape_file, 'rb') as f:
        line=f.readline().decode('ascii')
        if line.startswith('#'):
            shape=tuple(map(int, re.findall(r'(\d+)', line)))
            return shape
        else:
            raise IOError('Failed to find shape in file') 
