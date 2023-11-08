# SDXL.py
from diffusers import DiffusionPipeline
import torch
import io

def generate_image(prompt):
    # 모델 초기화
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    # 이미지 생성
    images = pipe(prompt=prompt).images[0]

    # 이미지를 바이트 스트림으로 변환하여 반환
    image_stream = io.BytesIO()
    images.save(image_stream, format='PNG')
    image_stream.seek(0)  # 스트림의 시작으로 포인터를 이동
    return image_stream
