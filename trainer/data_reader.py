import numpy as np
import os
import scipy.misc as misc
import random


class DataReader:

    def __init__(self, ImageDir, Label_Dir = None, batch_size = 1, shuffle = True):
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pointer = 0
        
        self.Image_Dir = ImageDir
        self.Label_Dir = Label_Dir
        
        if Label_Dir is None:
            self.ReadLabels=False
        else:
            self.ReadLabels=True
            
        
        self.Images = []
        
        if self.ReadLabels:
            self.Images += [each for each in os.listdir(self.Image_Dir)
            if ((each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith('.png') 
            or each.endswith('.jpg') or each.endswith('.tif')) 
            and os.path.isfile(self.Label_Dir + "/" + each[0:-4]+".png"))]
        else:
            self.Images += [each for each in os.listdir(self.Image_Dir)
            if each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith('.png') 
            or each.endswith('.jpg') or each.endswith('.tif') ]
        
    
        self.data_size = len(self.Images)
        
        if self.shuffle:
            self.shuffle_data()
        

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.Images.copy()
        self.Images = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(images))
        for i in idx:
            self.Images.append(images[i])
            
            
            
    def reset_pointer(self):
        """
        Reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
            
            
            
    def next_batch(self):
        
        images = self.Images[self.pointer:self.pointer + self.batch_size]
        
        self.pointer += self.batch_size
        
        
        result_images = []
        result_labels = []
        
        for i in range(len(images)):
            
            Img = misc.imread(self.Image_Dir + "/" + images[i])
            Img = Img[:,:,0:3]
            
            if self.ReadLabels:
                Label= misc.imread(self.Label_Dir + "/" + images[i][0:-4]+".png")
            
            #Random Mirror
            if random.random() < 0.5:
                Img=np.fliplr(Img)
                if self.ReadLabels:
                    Label=np.fliplr(Label)
                       
                    
            """
            For any image size (no resizing of the images)
            """
            """   
            result_images.append(Img)
            result_labels.append(Label[:,:,0])
            """
            
            if i==0:
                Sy,Sx,Depth=Img.shape
                result_images = np.zeros([self.batch_size,Sy,Sx,3], dtype=np.float)
                if self.ReadLabels: 
                    result_labels= np.zeros([self.batch_size,Sy,Sx], dtype=np.int)

            Img = misc.imresize(Img, [Sy, Sx], interp='bilinear')
            if self.ReadLabels: 
               Label = misc.imresize(Label, [Sy, Sx], interp='nearest')
           
            result_images[i] = Img
            if self.ReadLabels:
              result_labels[i] = Label[:,:,0]
              
        if self.ReadLabels:
               return np.array(result_images), np.array(result_labels)
        else:
               return np.array(result_images)
        
        
        
        
        