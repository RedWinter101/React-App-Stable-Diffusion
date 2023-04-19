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
@app.get("/")
def generate(prompt: str, Nprompt: str, inference: int, width: int, height: int): 
    with autocast(device): 
        image = pipe(prompt, negative_prompt=Nprompt, width=width, height=height, guidance_scale=8.5, num_inference_steps=inference, num_image=3).images[0]
    image.show()
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
