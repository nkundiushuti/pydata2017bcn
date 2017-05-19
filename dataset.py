"""
    This file is based on DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """
import numpy as np
import scipy
import os,sys
import util
import sklearn
from sklearn.model_selection import train_test_split

class MyDataset(object):
    """
    The class to load data in chunks and prepare batches for training neural networks

    Parameters
    ----------
    feature_dir : string
        The path for the folder from where to load the input data to the network
    batch_size : int, optional
        The number of examples in a batch
    time_context : int, optional
        The time context modeled by the network. 
        The data files are split into segments of this size
    step : int, optional
        The number of stepping frames between adjacent segments
    floatX: dtype
        Type of the arrays for the output
    """
    def __init__(self, feature_dir=None, batch_size=32, time_context=100, step=25, 
        suffix_in='_mel_', suffix_out='_label_', floatX=np.float32, train_percent=1.):
    
        self.batch_size = batch_size
        self.floatX = floatX
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        
        self.time_context = int(time_context)
        if step > self.time_context:
            self.step = int(0.5 * self.time_context)
        else:
            self.step = step

        self.train_percent = np.maximum(0.1,np.minimum(1.,train_percent))

        if feature_dir is not None:
            self.initDir(feature_dir)


    def getNumInstances(self,infile,time_context=100,step=25):
        """
        For a single .data file computes the number of examples of size \"time_context\" that can be created
        """
        shape = util.get_shape(os.path.join(infile.replace('.data','.shape')))
        length_file = float(shape[0])
        return np.maximum(1,int(np.ceil((length_file-time_context)/ self.step)))


    def getFeatureSize(self,infile):
        """
        For a single .data file return the number of feature, e.g. number of spectrogram bins
        """
        shape = util.get_shape(os.path.join(infile.replace('.data','.shape')))
        return shape[1]


    def initDir(self,feature_dir):        
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        #list of .data file names that contain the features
        self.file_list = [f for f in os.listdir(feature_dir) if f.endswith(self.suffix_in+'.data') and 
            os.path.isfile(os.path.join(feature_dir,f.replace(self.suffix_in,self.suffix_out))) ]
        self.total_files = len(self.file_list)
        assert self.total_files>0, "there are no feature files in the input directory"
        self.feature_dir = feature_dir

        #how many training examples we create for every file?
        #noinstances = self.getNumInstances(os.path.join(self.feature_dir,self.file_list[0]),time_context=self.time_context,step=self.step)
        self.total_noinstances = np.cumsum(np.array([0]+[self.getNumInstances(os.path.join(self.feature_dir,infile),time_context=self.time_context,step=self.step) 
            for infile in self.file_list], dtype=int))
        self.total_points = self.total_noinstances[-1]
        #reduce the batch size if we have less points
        self.batch_size=np.minimum(self.batch_size,self.total_points)
        #how many batches can we fit in the dataset 
        self.iteration_size=int(np.floor(self.total_points/ self.batch_size))
        self.iteration_step = -1

        #feature size (last dimension of the output)
        self.feature_size = self.getFeatureSize(infile=os.path.join(self.feature_dir,self.file_list[0]))

        #init the output 
        self.features=np.zeros((self.total_points,self.time_context,self.feature_size),dtype=self.floatX)
        self.labels=np.zeros((self.total_points,1),dtype=self.floatX)

        #fetch all data from hard-disk
        for id in range(self.total_files):
            self.fetchFile(id) 
        self.shuffleBatches()

        if self.train_percent<1.:       
            self.features_valid, self.features, self.labels_valid, self.labels= train_test_split(self.features, self.labels, test_size=self.train_percent, random_state=42)
            self.total_points = len(self.features)
            self.iteration_size = int(np.floor(self.total_points/ self.batch_size))


    def shuffleBatches(self):
        idxrand = np.random.permutation(self.total_points)
        self.features=self.features[idxrand]
        self.labels=self.labels[idxrand]


    def fetchFile(self,id):
        #load the data files
        spec = util.loadTensor(out_path=os.path.join(self.feature_dir,self.file_list[id]))
        lab = util.loadTensor(out_path=os.path.join(self.feature_dir,self.file_list[id].replace(self.suffix_in,self.suffix_out)))
        #we need to put the features in the self.features array starting at this index
        idx_start = self.total_noinstances[id]
        #and we stop at this index in self.feature
        idx_end = self.total_noinstances[id+1]
  
        #copy each block of size (time_contex,feature_size) in the self.features
        idx=0 #starting index of each block
        start = 0 #starting point for each block in frames
        while idx<(idx_end-idx_start):
            self.features[idx_start+idx] = spec[start:start+self.time_context]
            start = start + self.step 
            idx = idx + 1
        self.labels[idx_start:idx_end] = lab[0]
        spec = None
        lab = None
 

    def getData(self):
        assert self.total_points>0, "no data points in dataset"
        if self.train_percent<1.:
            return self.features,self.labels, self.features_valid, self.labels_valid 
        else:
            return self.features,self.labels, self.features,self.labels

    def getValidation(self):
        assert self.train_percent<1., "no validation examples. decrease train_percent"
        return self.features_valid, self.labels_valid 

    def iterate(self):
        self.iteration_step += 1
        if self.iteration_step==self.iteration_size:
            self.iteration_step = 0
            self.shuffleBatches()
            y = self.labels[self.iteration_step*self.batch_size:(self.iteration_step+1)*self.batch_size]
            categorical = np.zeros(self.batch_size, self.labels.max()+1)
            y = np.array(y, dtype='int').ravel()
            categorical[np.arange(self.batch_size), y] = 1

        return self.features[self.iteration_step*self.batch_size:(self.iteration_step+1)*self.batch_size,np.newaxis],\
               categorical

    def __len__(self):
        return self.iteration_size

    def __call__(self):
        return self.iterate()

    def __iter__(self):
        return self.iterate()

    def next(self):
        return self.iterate()

    def batches(self):
        return self.iterate()
