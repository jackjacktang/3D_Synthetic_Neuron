set -ex
python ../train.py --dataroot ../datasets/datasets/edges2shoes --name shoes_pix2pix --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0
