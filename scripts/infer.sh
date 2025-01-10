#!/bin/bash

# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=weights/infinity_2b_reg.pth
vae_type=32
vae_path=weights/infinity_vae_d32reg.pth
cfg=4
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=weights/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0

# run inference
python3 tools/run_infinity.py \
--cfg ${cfg} \
--tau ${tau} \
--bf16 1 \
--pn ${pn} \
--model_path ${infinity_model_path} \
--vae_type ${vae_type} \
--vae_path ${vae_path} \
--add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
--use_bit_label ${use_bit_label} \
--model_type ${model_type} \
--rope2d_each_sa_layer ${rope2d_each_sa_layer} \
--rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
--use_scale_schedule_embedding ${use_scale_schedule_embedding} \
--cfg ${cfg} \
--tau ${tau} \
--checkpoint_type ${checkpoint_type} \
--text_encoder_ckpt ${text_encoder_ckpt} \
--text_channels ${text_channels} \
--apply_spatial_patchify ${apply_spatial_patchify} \
--prompt "a tornado made out of cake in the style of mad max" \
--seed 1 \
--save_file tmp.jpg
