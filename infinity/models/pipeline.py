import torch
from typing import List, Union, Optional
import os
import os.path as osp
import cv2
from transformers import AutoTokenizer, T5EncoderModel
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

from infinity.models.infinity import Infinity, sample_with_top_k_top_p_also_inplace_modifying_logits_
from infinity.models.basic import CrossAttnBlock
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from infinity.models.bsq_vae.vae import vae_model
from tools.run_infinity import load_transformer, load_visual_tokenizer, load_tokenizer

class InfinityPipeline:
    """Pipeline for text-to-image generation using Infinity model."""
    
    def __init__(
        self,
        infinity_model: Infinity,
        vae: vae_model,
        text_tokenizer: AutoTokenizer,
        text_encoder: T5EncoderModel,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.infinity = infinity_model
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = torch_dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        text_encoder_path: str = "google/flan-t5-xl",
        vae_path: str = None,
        pn: str = "1M",
        model_type: str = "infinity_2b",
        **kwargs
    ):
        """Load pretrained pipeline components."""
        
        # Create args namespace to match original implementation
        from argparse import Namespace

        vae_types = {
            "infinity_2b": 32,
            "infinity_8b": 14,
        }
        args = Namespace(
            pn=pn,
            model_path=pretrained_model_name_or_path,
            cfg_insertion_layer=0,
            vae_type=vae_types[model_type],
            vae_path=vae_path,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type=model_type,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt=text_encoder_path,
            text_channels=2048,
            apply_spatial_patchify=0 if model_type == "infinity_2b" else 1,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir='/dev/shm',
            enable_model_cache=0,
            checkpoint_type='torch_shard' if osp.isdir(pretrained_model_name_or_path) else 'torch',
            seed=0,
            bf16=1 if torch_dtype == torch.bfloat16 else 0
        )

        # Load components using original functions
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        vae = load_visual_tokenizer(args)
        infinity = load_transformer(vae, args)
        
        return cls(
            infinity_model=infinity,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            device=device,
            torch_dtype=torch_dtype,
        )
        
    def encode_prompt(self, prompt: Union[str, List[str]], enable_positive_prompt=False):
        """Encode text prompt into embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]
            
        tokens = self.text_tokenizer(
            text=prompt,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(self.device)
        
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=mask
        )['last_hidden_state']
        
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)
        
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        
        kv_compact = torch.cat(kv_compact, dim=0)
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
        return text_cond_tuple

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        cfg_scale: float = 3.0,
        tau: float = 0.5,
        top_k: int = 900,
        top_p: float = 0.97,
        h_div_w: float = 1.0,
        pn: str = "1M",
        **kwargs
    ):
        """Generate images from text prompt."""
        
        # Handle prompt batching
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        
        device = self.device
        
        # Get scale schedule based on resolution
        scale_schedule = dynamic_resolution_h_w[h_div_w][pn]["scales"]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        
        if self.infinity.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        # Encode prompts including negative prompt
        text_cond_tuple = self.encode_prompt(prompt)
        
        cfg_list = [cfg_scale] * len(scale_schedule)
        tau_list = [tau] * len(scale_schedule)

        # Generate images
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            with torch.no_grad():
                # Initialize RNG if seed provided
                if seed is None:
                    rng = None
                else:
                    rng = torch.Generator(device=device).manual_seed(seed)

                # Prepare text conditioning
                kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
                
                # Handle CFG (Classifier Free Guidance)
                if cfg_scale != 1:
                    bs = 2 * batch_size
                    kv_compact_un = kv_compact.clone()
                    total = 0
                    for le in lens:
                        kv_compact_un[total:total+le] = self.infinity.cfg_uncond[:le]
                        total += le
                    kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                    cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
                else:
                    bs = batch_size

                # Process text embeddings
                kv_compact = self.infinity.text_norm(kv_compact)
                sos = cond_BD = self.infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))
                kv_compact = self.infinity.text_proj_for_ca(kv_compact)
                ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
                last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.infinity.pos_start.expand(bs, 1, -1)

                # Initialize adaptive layer norm
                cond_BD_or_gss = self.infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
                
                # Initialize tracking variables
                summed_codes = 0

                # Enable KV caching for inference
                for b in self.infinity.unregistered_blocks:
                    (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

                # Process each scale in schedule
                num_stages_minus_1 = len(scale_schedule)-1
                cur_L = 0
                
                for si, pn in enumerate(scale_schedule):
                    if si >= 1000:  # trunk_scale
                        break
                        
                    cur_L += np.array(pn).prod()
                    
                    # Get attention function if using flex attention
                    attn_fn = None
                    if self.infinity.use_flex_attn:
                        attn_fn = self.infinity.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

                    # Process through transformer blocks
                    layer_idx = 0
                    for block_idx, b in enumerate(self.infinity.block_chunks):
                        if self.infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                            last_stage = self.infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
                        if not self.infinity.add_lvl_embeding_only_first_block:
                            last_stage = self.infinity.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=0)
                        
                        for m in b.module:
                            last_stage = m(
                                x=last_stage, 
                                cond_BD=cond_BD_or_gss,
                                ca_kv=ca_kv,
                                attn_bias_or_two_vector=None,
                                attn_fn=attn_fn,
                                scale_schedule=scale_schedule,
                                rope2d_freqs_grid=self.infinity.rope2d_freqs_grid,
                                scale_ind=si
                            )
                            if (cfg_scale != 1) and (layer_idx == 0):  # cfg_insertion_layer=0
                                last_stage = cfg_scale * last_stage[:batch_size] + (1-cfg_scale) * last_stage[batch_size:]
                                last_stage = torch.cat((last_stage, last_stage), 0)
                            layer_idx += 1

                    # Get logits and sample
                    logits_BlV = self.infinity.get_logits(last_stage[:batch_size], cond_BD[:batch_size]).mul(1/tau_list[si])
                    
                    # Handle bit label sampling
                    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                    idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                        logits_BlV, 
                        rng=rng,
                        top_k=top_k,
                        top_p=top_p,
                        num_samples=1
                    )[:, :, 0]
                    idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)

                    # Process VAE codes
                    idx_Bld = idx_Bld.reshape(batch_size, pn[1], pn[2], -1)
                    if self.infinity.apply_spatial_patchify:
                        idx_Bld = idx_Bld.permute(0,3,1,2)
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)
                        idx_Bld = idx_Bld.permute(0,2,3,1)
                    idx_Bld = idx_Bld.unsqueeze(1)

                    codes = self.vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')
                    
                    if si != num_stages_minus_1:
                        summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up)
                        last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=self.vae.quantizer.z_interplote_up)
                        last_stage = last_stage.squeeze(-3)
                        if self.infinity.apply_spatial_patchify:
                            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)
                        last_stage = last_stage.reshape(*last_stage.shape[:2], -1)
                        last_stage = torch.permute(last_stage, [0,2,1])
                    else:
                        summed_codes += codes

                    if si != num_stages_minus_1:
                        last_stage = self.infinity.word_embed(self.infinity.norm0_ve(last_stage))
                        last_stage = last_stage.repeat(bs//batch_size, 1, 1)

                # Disable KV caching
                for b in self.infinity.unregistered_blocks:
                    (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

                # Decode final image
                img = self.vae.decode(summed_codes.squeeze(-3))
                img = (img + 1) / 2
                img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
                
                return img
