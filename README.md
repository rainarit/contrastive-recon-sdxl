# CSE 527 Final Project: Self-supervised Contrastive Learning Framework for Satellite Image Representations

This coding implementation is inspired from the MAE repo https://github.com/facebookresearch/mae, and the CAN repo https://github.com/shlokk/mae-contrastive.

## Important files to keep in mind
- `main_pretrain.py` : python script for pre-training the model by utilizing contrastive and reconstruction loss.
- `main_linprobe.py` : python script for linear probing by utilizing a linear classifier at teh head of the model for classifying each of the 45 scenes.
- `generate_imgs.py` : python script that employs Stable Diffusion XL for generating each of the images for RESISC-45-SDXL.

## Generating synthetic image dataset (RESISC-45-SDXL)
- You must first install the following packages first.
```
pip install diffusers
pip install accelerate
pip install huggingface
```
- Run the `generate_imgs.py` file.
```
accelerate launch generate_imgs.py
```

## Requirements to do before running any experiments
- Creating the conda environment in order to run the appropriate experiments <br>

```
conda create -n contrastive-recon-sdxl python=3.8 -y
source activate contrastive-recon-sdxl
pip install -r requirements.txt
```

## Script for running model pre-training
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --num_workers 8 --blr 1e-4 --weight_decay 0.05 --model mae_vit_base_patch16 \
    --data_path path_to_your_dataset \
    --batch_size 64 --epochs 40 --weight_simclr 0.03 --accum_iter 4 \
    --finetune /home/rraina/mae-contrastive/mae_pretrain_vit_base.pth \
    --output_dir path_to_your_output_dir \
    --log_dir path_to_your_log_dir
```


## Script for running linear probing:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py --model vit_base_patch16 \
--cls_token --finetune path_to_trained_model \
--data_path path_to_your_dataset \
--output_dir path_to_your_output_dir \
--log_dir path_to_your_log_dir \
--epochs 50 --blr 1.0 --dist_eval 
```

** Disclaimer: We used a pre-trained masked ViT-B encoder when we did model pre-training (you can find the checkpoint weights over here: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)
