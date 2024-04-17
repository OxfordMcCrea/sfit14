import sys
import numpy as np

class array_functions:
    '''
    An object with some logic to hold future arrays:\n
    Arrays in dictionary a have all functions applied to them
    '''
    def __init__(self):
        self.a = {}
        self.gtraj = 0
        self.percent = 0

    def reshape_tall(self):
        '''Turn [1,n] arrays into [n,1] arrays'''
        for key in self.a.keys():
            array = self.a[key]
            if len(array.shape) != 1: continue
            self.a[key] = np.reshape(array,(len(array),1))

    def reshape_flat(self):
        '''Turn [n,1] arrays into [1,n] arrays'''
        for key in self.a.keys():
            array = self.a[key]
            if len(array.shape) != 1: continue
            self.a[key] = np.reshape(array,(1,len(array)))

    def shorten_arrays(self,bool,ntraj):
        '''Shorten all arrays using booleans'''
        #reshape bool to be flat
        bool = np.reshape(bool,(1,len(bool)))[0]
        for key in self.a.keys():
            array = self.a[key]
            if len(array) == 1 and bool == False:
                print('No successful trajectories remain, exiting.')
                sys.exit()
            #If array is tall
            try: self.a[key] = array[bool,:]
            #If array is flat
            except IndexError: self.a[key] = array[bool]
        self.reshape_tall()
        self.gtraj = len(self.a['V_NO_trajx'])
        self.percent = round((self.gtraj / ntraj) * 100,2)