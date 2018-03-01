
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from data import dataProcess
from keras.preprocessing.image import array_to_img

#
import Unet, Unet2

unet=Unet.get_unet
#unet=Unet2.get_unet

import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

		return unet(self.img_rows, self.img_cols)

	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, \
		          verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('./results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		imgs = np.load('./results/imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("./results/%d.jpg"%(i))

if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	myunet.save_img()








