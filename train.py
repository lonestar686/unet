
import time
import argparse
#
from nets.MyNet import myNet

if __name__ == '__main__':
	# get the parameters
	parser=argparse.ArgumentParser()
	parser.add_argument("--niters", type=int, default=20, help="number of epochs")
	parser.add_argument("--batches", type=int, default=4, help="batch sizes")
	args=parser.parse_args()

	#
	print('start time: {}'.format(time.ctime()))
	mynet = myNet()
	mynet.train_and_predict(n_epochs=args.niters, n_bsize=args.batches)
	print("end time: {}".format(time.ctime()))
