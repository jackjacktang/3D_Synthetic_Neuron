set -ex
python ../train.py --dataroot ../datasets/datasets/fly/fly2d --name 2d_neuron_pix2pix --checkpoints_dir ../checkpoints --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode neuron --input_nc 1 --output_nc 1 --norm batch --pool_size 0
