# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import argparse
import subprocess
import time
from cog import BasePredictor, Input, Path
import torch
import cv2
import numpy as np
from tools.run_infinity import (
    load_tokenizer,
    load_infinity,
    load_visual_tokenizer,
    gen_one_img,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

MODEL_CACHE = "model_cache"
MODEL_URL = f"https://weights.replicate.delivery/default/FoundationVision/Infinity/model_cache.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def load_transformer(vae, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path

    # Define model configuration based on type
    model_configurations = {
        "infinity_2b": dict(
            depth=32,
            embed_dim=2048,
            num_heads=2048 // 128,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=8,
        ),
        "infinity_layer12": dict(
            depth=12,
            embed_dim=768,
            num_heads=8,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=4,
        ),
        "infinity_layer16": dict(
            depth=16,
            embed_dim=1152,
            num_heads=12,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=4,
        ),
        "infinity_layer24": dict(
            depth=24,
            embed_dim=1536,
            num_heads=16,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=4,
        ),
        "infinity_layer32": dict(
            depth=32,
            embed_dim=2080,
            num_heads=20,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=4,
        ),
        "infinity_layer40": dict(
            depth=40,
            embed_dim=2688,
            num_heads=24,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=4,
        ),
        "infinity_layer48": dict(
            depth=48,
            embed_dim=3360,
            num_heads=28,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=4,
        ),
    }

    kwargs_model = model_configurations.get(args.model_type, {})
    if not kwargs_model:
        raise ValueError(f"Unknown model type: {args.model_type}")

    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=model_path,  # Directly use model_path
        scale_schedule=None,
        vae=vae,
        device=device,
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
    )
    return infinity


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        model_path = f"{MODEL_CACHE}/FoundationVision/Infinity/infinity_2b_reg.pth"
        vae_path = f"{MODEL_CACHE}/FoundationVision/Infinity/infinity_vae_d32reg.pth"
        text_encoder_ckpt = f"{MODEL_CACHE}/google/flan-t5-xl"
        self.args = argparse.Namespace(
            pn="1M",
            model_path=model_path,
            cfg_insertion_layer=0,
            vae_type=32,
            vae_path=vae_path,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type="infinity_2b",
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt=text_encoder_ckpt,
            text_channels=2048,
            apply_spatial_patchify=0,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/tmp/cache",
            checkpoint_type="torch",
            bf16=1,
        )

        self.text_tokenizer, self.text_encoder = load_tokenizer(
            t5_path=text_encoder_ckpt
        )
        # load vae
        self.vae = load_visual_tokenizer(self.args)
        # load infinity
        self.infinity = load_transformer(self.vae, self.args)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="alien spaceship enterprise",
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=10, default=3
        ),
        tau: float = Input(description="tau in self attention", default=0.5),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        h_div_w = 1 / 1  # aspect ratio, height:width
        h_div_w_template_ = h_div_w_templates[
            np.argmin(np.abs(h_div_w_templates - h_div_w))
        ]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn][
            "scales"
        ]
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        generated_image = gen_one_img(
            self.infinity,
            self.vae,
            self.text_tokenizer,
            self.text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=guidance_scale,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer],
            vae_type=self.args.vae_type,
            sampling_per_bits=self.args.sampling_per_bits,
            enable_positive_prompt=0,
        )
        output_path = "/tmp/out.png"
        cv2.imwrite(output_path, generated_image.cpu().numpy())
        return Path(output_path)
