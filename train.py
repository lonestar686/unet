
import time
import argparse

# pick cpu/gpu
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # CPU

# Unet
#from nets.Unet import Unet as Net

# Unet2
from nets.Unet_kaggle import Unet_Kaggle as Net

# Unet3
#from nets.Unet_bn import Unet_BN as Net

# linknet
#from nets.Linknet import Linknet as Net

# linknet
#from nets.Linknet_s import Linknet_s as Net

# Tiramisu
#from nets.Tiramisu import Tiramisu as Net

# get the parameters
parser=argparse.ArgumentParser(description='for image semantic segmentation')
parser.add_argument("--niters", type=int, default=20, help="number of epochs")
parser.add_argument("--batches", type=int, default=4, help="batch sizes")

if __name__ == '__main__':
	# get commandline arguments
	args=parser.parse_args()

	#
	print('start time: {}'.format(time.ctime()))
	mynet = Net(img_rows = 512, img_cols = 512, growth_rate=8, nb_filter=48, nb_layers_per_block=2)
	mynet.train_and_predict(n_epochs=args.niters, n_bsize=args.batches)
	print("end time: {}".format(time.ctime()))
