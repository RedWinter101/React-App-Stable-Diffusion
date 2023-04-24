!pip install "uvicorn[standard]"
!pip install fastapi nest-asyncio pyngrok
!pip install pybase64 
!pip install Pillow
!pip install diffusers  
!pip install pip install git+https://github.com/huggingface/transformers
!pip install scipy ftfy accelerate
!pip install torch
!pip install python-multipart

from fastapi import FastAPI, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO
import base64 
from PIL import Image
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

from fastapi import FastAPI, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO
import base64 
from PIL import Image
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

from huggingface_hub import upload_folder
device = "cuda"
model_id = "SG161222/Realistic_Vision_V1.3"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision="main", torch_dtype=torch.float16)
pipe.to(device)

from google.colab import drive
drive.mount('/content/gdrive')

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

@app.post("/")
def upload_file(upload: UploadFile):
  uploadf = upload
  return uploadf

@app.get("/")
def generate(prompt: str, Nprompt: str, inference: int, cfg_scale: float, width: int, height: int):
    url = "https://raw.githubusercontent.com/RedWinter101/React-App-Stable-Diffusion/main/api/Mountain-10.png"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((width, height))

    with autocast(device): 
        image = pipe(prompt, image=init_image, strength=cfg_scale, num_images_per_prompt=2, negative_prompt=Nprompt, guidance_scale=8.5, num_inference_steps=inference).images
    init_image.show()
    grid = image_grid(image, rows=1, cols=2)
    grid.show()
    
    grid.save(f"test.png")
    buffer = BytesIO()
    grid.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")
  
import nest_asyncio
from pyngrok import ngrok
import uvicorn
!ngrok config add-authtoken 2OOjnLNtbhW4HOD69RcZtMloWbP_3Jm6hbLi3DN2StK9dZK37
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
