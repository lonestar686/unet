
import os

#
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from data import dataProcess
from keras.preprocessing.image import array_to_img
from keras.optimizers import Adam

#
from CustomLosses import gumble_loss

# Unet
#import Unet
#net=Unet.get_net

# Unet2
import Unet2
net=Unet2.get_net

# Unet3
#import Unet3
#net=Unet3.get_net

# linknet
#import linknet
#net=linknet.LinkNet

#
n_epochs = 10

#
class myNet(object):

	def __init__(self, img_rows = 512, img_cols = 512, out_dir='./results', model_dir='./model'):

		self.img_rows = img_rows
		self.img_cols = img_cols

		# directories
		self.out_dir = out_dir
		self.model_dir = model_dir
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
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    	#model.compile(optimizer = Adam(lr = 1e-4), loss = gumble_loss, metrics = ['accuracy'])

        #
		model_checkpoint = ModelCheckpoint(self.model_name(), monitor='loss',\
                                           verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=n_epochs, \
		          verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
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
