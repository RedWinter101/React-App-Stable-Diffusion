from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = "cuda"
model_id = "SG161222/Realistic_Vision_V1.3"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="main", torch_dtype=torch.float16)
pipe.to(device)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

@app.get("/")
def generate(prompt: str, Nprompt: str, inference: int, width: int, height: int):
    with autocast(device): 
        image = pipe(prompt, num_images_per_prompt=2, negative_prompt=Nprompt, width=width, height=height, guidance_scale=8.5, num_inference_steps=inference).images
    
    grid = image_grid(image, rows=1, cols=2)
    grid.show()
    
    grid.save(f"test.png")
    buffer = BytesIO()
    grid.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
