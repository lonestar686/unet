
import time

#
from MyNet import myNet

if __name__ == '__main__':
	#
	print('start time: {}'.format(time.ctime()))
	mynet = myNet()
	mynet.train_and_predict()
	print("end time: {}".format(time.ctime()))
