#install in google colab

!pip install "uvicorn[standard]"
!pip install fastapi nest-asyncio pyngrok
!pip install pybase64 
!pip install Pillow
!pip install diffusers  
!pip install pip install git+https://github.com/huggingface/transformers
!pip install scipy ftfy accelerate
!pip install torch
!pip install python-multipart
!pip install opencv-contrib-python
!pip install diffusers transformers git+https://github.com/huggingface/accelerate.git


###########################################

from fastapi import FastAPI, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from huggingface_hub import upload_folder
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel
from io import BytesIO
import base64 
from PIL import Image
import requests
import cv2
import numpy as np

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
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

pipe1 = StableDiffusionPipeline.from_pretrained(model_id, revision="main", torch_dtype=torch.float16)

pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision="main", torch_dtype=torch.float16)

pipe3 = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16)

pipe1.to(device)
pipe2.to(device)
pipe3.to(device)
generator = torch.manual_seed(0)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


@app.get("/")
def generate_txt2img(prompt: str, Nprompt: str, inference: int, width: int, height: int):
    with autocast(device): 
        image = pipe1(prompt, num_images_per_prompt=2, negative_prompt=Nprompt, 
                     width=width, height=height, guidance_scale=8.5, num_inference_steps=inference).images
    
    grid = image_grid(image, rows=1, cols=2)
    grid.show()


@app.post("/img2img")
async def generate_img2img(prompt: str, Nprompt: str, inference: int, cfg_scale: float, width: int, height: int, file: UploadFile = File(...)):

    file_b = await file.read()
    imgs = Image.open(BytesIO(file_b))
    imgs.show()

    image = np.array(imgs)

    with autocast(device): 
        image = pipe2(prompt, image=image, strength=cfg_scale, num_images_per_prompt=2, 
                     negative_prompt=Nprompt, guidance_scale=8.5, num_inference_steps=inference).images
    imgs.show()
    grid = image_grid(image, rows=1, cols=2)
    grid.show()


@app.post("/controlnet")
async def generate_controlnet(prompt: str, Nprompt: str, inference: int, width: int, height: int, file: UploadFile = File(...)):

    file_b = await file.read()
    imgs = Image.open(BytesIO(file_b))
    imgs.show()

    image = np.array(imgs)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    with autocast(device): 
        image = pipe3(prompt, image=canny_image, num_images_per_prompt=2, 
                     negative_prompt=Nprompt,  eta=0.5, guidance_scale=8.5, num_inference_steps=inference, 
                     width=width, height=height, generator=generator,
                     controlnet_conditioning_scale=1.00).images
                     
    canny_image.show()
    imgs.show()
    grid = image_grid(image, rows=1, cols=2)
    grid.show()
  

import nest_asyncio
from pyngrok import ngrok
import uvicorn
!ngrok config add-authtoken #NGROK Token
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
