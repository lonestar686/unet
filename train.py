
# importing the libraries
#import numpy as np
#import tensorflow as tf
#import random as rn

#import os
#os.environ['PYTHONHASHSEED'] = '0'

#from keras import backend as k

# Running the below code every time
#np.random.seed(27)
#rn.seed(27)
#tf.set_random_seed(27)

#sess = tf.Session(graph=tf.get_default_graph())
#k.set_session(sess)


# pick cpu/gpu
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time

#
from MyNet import myNet


if __name__ == '__main__':
	#
	print('start time: {}'.format(time.ctime()))
	mynet = myNet()
	mynet.train_and_predict()
	print("end time: {}".format(time.ctime()))
