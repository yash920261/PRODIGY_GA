import torch
from diffusers import StableDiffusionPipeline
import os

hf_token = ""  

prompt = "a futuristic city on Mars at sunset, ultra-detailed, cinematic lighting"
output_file = "generated_image.png"


device = "cuda" if torch.cuda.is_available() else "cpu"


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=hf_token  
)
pipe = pipe.to(device)


with torch.autocast(device) if device == "cuda" else torch.no_grad():
    image = pipe(prompt).images[0]

image.save(output_file)
print(f"Image saved to: {output_file}")
