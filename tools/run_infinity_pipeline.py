import os
import time
import torch
import cv2
from datetime import datetime
from infinity.models.pipeline import InfinityPipeline

def save_images(images, output_dir, prompt_prefix="image"):
    """Save a list of images with timestamp."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_paths = []
    for i, image in enumerate(images):
        filename = f"{prompt_prefix}_{timestamp}_{i}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, image.cpu().numpy())
        saved_paths.append(save_path)
        print(f"Saved image to: {save_path}")
    
    return saved_paths

def main():
    # Model paths
    model_path = "weights/infinity_2b_reg.pth"
    vae_path = "weights/infinity_vae_d32_reg.pth"
    text_encoder_path = "weights/flan-t5-xl"
     
    # Initialize pipeline
    print("Initializing Infinity Pipeline...")
    pipe = InfinityPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        model_type="infinity_2b",
        device="cuda",
        torch_dtype=torch.bfloat16,
        pn="1M"
    )
    
    # Example prompts
    prompts = [
        "A majestic dragon made of crystal",
        "A close-up photograph of a Corgi dog",
        "A photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest with the text \'Welcome Friends!"
    ]
    
    print(f"\nGenerating images for prompts...")
    start_time = time.time()
    
    # Generate images
    images = pipe(
        prompt=prompts,
        cfg_scale=3.0,
        tau=0.5,
        seed=42,
        top_k=900,
        top_p=0.97
    )
    
    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    # Save the generated images
    save_images(images, "outputs", "batch_infers")

if __name__ == "__main__":
    main()
