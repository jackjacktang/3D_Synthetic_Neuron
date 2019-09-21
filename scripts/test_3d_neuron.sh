set -ex

# basic 3d u-net
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_unet_neuron_z32 --checkpoints_dir ../checkpoints --model pix2pix --netG unet_3d_cust --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d

# 3d linknet
# python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_linknet_neuron --checkpoints_dir ../checkpoints --model pix2pix --netG linknet_3d --direction AtoB --dataset_mode neuron3d --input_nc 1 --output_nc 1 --crop_size_3d 128x128x32 --norm batch --results_dir ../checkpoints --save_type 3d