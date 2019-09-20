set -ex
python ../test.py --dataroot ../datasets/datasets/fly/fly3d/ --name 3d_unet_neuron --checkpoints_dir ../checkpoints --model pix2pix --netG unet_3d --direction AtoB --dataset_mode neuron --input_nc 1 --output_nc 1 --norm batch --results_dir ../checkpoints --save_type 3d