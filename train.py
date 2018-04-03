
import time
import argparse

# pick cpu/gpu
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # CPU

# Unet
#from nets.Unet import Net

# Unet2
from nets.Unet_kaggle import Net

# Unet3
#from nets.Unet_bn import Net

# linknet
#from nets.Linknet import Net

# Tiramisu
#from nets.Tiramisu import Net

if __name__ == '__main__':
	# get the parameters
	parser=argparse.ArgumentParser()
	parser.add_argument("--niters", type=int, default=20, help="number of epochs")
	parser.add_argument("--batches", type=int, default=4, help="batch sizes")
	args=parser.parse_args()

	#
	print('start time: {}'.format(time.ctime()))
	mynet = Net()
	mynet.train_and_predict(n_epochs=args.niters, n_bsize=args.batches)
	print("end time: {}".format(time.ctime()))
