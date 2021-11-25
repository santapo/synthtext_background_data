# import the necessary packages

from __future__ import division
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import cv2
import h5py
import numpy as np
import multiprocessing as mp
import traceback, sys
from collections import Counter
import sys

img_folder= 'test_images/'
output_file= sys.argv[2]
img=sys.argv[1] 
image = cv2.imread(img_folder+img)
segments = slic(img_as_float(image), n_segments = 30 , sigma = 5)
areas = []
labels=[]
s= segments.reshape(segments.shape[1]*segments.shape[0])
word_count = Counter(s)
occ=word_count.items()
for i in occ:
	areas.append(i[1])
	labels.append(i[0]+1)

# print np.array(areas) 
# print np.array(labels)

# output h5 file:
dbo = h5py.File(output_file,'a')
mask_dset = dbo.create_dataset('/seg/'+img, data=segments)
mask_dset.attrs['area'] = areas
mask_dset.attrs['label'] = labels
import ipdb; ipdb.set_trace()
dbo.close()
