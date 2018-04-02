
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: 
# https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# ---------------------------

# pick cpu/gpu
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import array_to_img
from keras.optimizers import Adam

#
from dataprep.data import dataProcess

#
from nets.CustomLosses import gumble_loss

# Unet
#from nets import Unet
#net=Unet.Net

# Unet2
from nets import Unet_kaggle
net=Unet_kaggle.Net

# Unet3
#from nets import Unet_bn
#net=Unet_bn.Net

# linknet
#from nets import Linknet
#net=Linknet.Net

#
n_epochs = 50
n_bsize = 2

#
class myNet(object):

	def __init__(self, img_rows = 512, img_cols = 512, out_dir='./results', model_dir='./model'):

		self.img_rows = img_rows
		self.img_cols = img_cols

		# directories
		self.out_dir = out_dir
		self.model_dir = model_dir
#
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)

	def load_train_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()                                            

		return imgs_train, imgs_mask_train

	def get_net(self):

		return net(self.img_rows, self.img_cols)

	def train_and_predict(self):

		print("loading data")
		imgs_train, imgs_mask_train = self.load_train_data()
		print("loading data done")

		model = self.get_net()
		print("got net")
		#
		model.summary()
        
		# for training
		#model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		model.compile(optimizer = Adam(lr = 1e-4), loss = gumble_loss, metrics = ['accuracy'])

        #
		model_checkpoint = ModelCheckpoint(self.model_name(), monitor='loss',\
                                           verbose=1, save_best_only=True)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, \
                              patience=20, min_lr=1e-6, verbose=1)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=n_bsize, epochs=n_epochs, \
		          verbose=1,validation_split=0.2, shuffle=True, \
				  callbacks=[model_checkpoint, reduce_lr])
		#
		self.predict(model)


	def predict(self, model):

		print('load test data')
		# load data
		mydata    = dataProcess(self.img_rows, self.img_cols)
		imgs_test = mydata.load_test_data()

		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save(self.test_result_name(), imgs_mask_test)

		print('save results to jpeg files')
		self.save_img()

	def test(self):

		print('reload model')
		# load model
		model = self.load_model()

		# predict
		self.predict(model)

	def load_model(self):
		print('reload model configuration and weight')
		model = self.get_net()
		model.load_weights(self.model_name())
		
		return model

	def model_name(self):
		return os.path.join(self.model_dir, "net.hd5")

	def test_result_name(self):
		return os.path.join(self.out_dir, 'imgs_mask_test.npy')
    
	def save_img(self):

		print("array to image")
		imgs = np.load(self.test_result_name())
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save(os.path.join(self.out_dir, "%d.jpg"%(i)))
