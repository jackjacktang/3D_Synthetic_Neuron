set -ex
python test.py --dataroot ../datasets/datasets/edges2shoes --name shoes_pix2pix --model pix2pix --netG unet_256 --direction BtoA --dataset_mode aligned --norm batch
