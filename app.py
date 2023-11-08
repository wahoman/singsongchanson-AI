from flask import Flask, request, jsonify, abort
import uuid
import boto3
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from audiocraft.models import musicgen, MultiBandDiffusion
from audiocraft.data.audio import audio_write
from lc_project import analyze_input
from SDXL import generate_image
from dotenv import load_dotenv
import os

# 환경 변수를 로드합니다.
load_dotenv()

app = Flask(__name__)

# BLIP 모델 초기화
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")



# 환경 변수 설정
AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

@app.route("/generate_music", methods=["POST"])
def generate_music():
    # 요청으로부터 데이터를 추출합니다.
    keyword = request.json.get("keyword")
    
    # 음악 생성에 필요한 설정 값을 정의합니다.
    model_name = "facebook/musicgen-large"
    duration = 30
    strategy = "loudness"
    sampling = True
    top_k = 0
    top_p = 0.9
    temperature = 0.9
    use_diffusion = False
    use_custom = False

    # 입력 분석 및 음악 생성 프롬프트를 설정합니다.
    analyzed_data = analyze_input(keyword)
    prompt = TEMPLATE.format(**analyzed_data)
    print("Music prompt: ", prompt)

    # 모델 이름, 지속 시간 등의 유효성을 검사합니다.
    validate_generation_parameters(model_name, duration, prompt, strategy, sampling, top_k, top_p, temperature, use_diffusion, use_custom)

    # 음악 생성 및 S3 업로드 로직을 실행합니다.
    s3_file_path_music = generate_and_upload_music(model_name, prompt, duration, sampling, top_k, top_p, temperature, use_diffusion, use_custom)
    
    # 이미지 생성 프롬프트를 생성합니다.
    image_prompt = f"An k-pop music album cover-style photo with an abstract design for a {keyword}."

    # 이미지 생성 및 S3 업로드 로직을 실행합니다.
    s3_file_path_image = generate_and_upload_image(image_prompt)

    # 생성된 음악과 이미지의 S3 URL을 반환합니다.
    response = {
        "result": "success",
        "music_url": f"https://{BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/{s3_file_path_music}",
        "image_url": f"https://{BUCKET_NAME}.s3.ap-northeast-2.amazonaws.com/{s3_file_path_image}"
    }

    return jsonify(response)

# 유효성 검사를 수행하는 함수
def validate_generation_parameters(model_name, duration, prompt, strategy, sampling, top_k, top_p, temperature, use_diffusion, use_custom):
    if model_name not in ["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"]:
        abort(400, "Invalid model name")
    
    if duration not in [15, 30, 60, 90, 120, 180]:
        abort(400, "Invalid duration")
    
    if not isinstance(prompt, str):
        abort(400, "Invalid prompt type")
    
    if strategy not in ["loudness", "peak", "clip"]:
        abort(400, "Invalid strategy")
    
    if not isinstance(sampling, bool):
        abort(400, "Invalid sampling type")
    
    if not isinstance(top_k, int):
        abort(400, "Invalid top_k type")
    
    if not isinstance(top_p, float):
        abort(400, "Invalid top_p type")
    
    if not isinstance(temperature, float):
        abort(400, "Invalid temperature type")
    
    if not isinstance(use_diffusion, bool):
        abort(400, "Invalid use_diffusion type")
    
    if not isinstance(use_custom, bool):
        abort(400, "Invalid use_custom type")

# 음악을 생성하고 S3에 업로드하는 함수
def generate_and_upload_music(model_name, prompt, duration, sampling, top_k, top_p, temperature, use_diffusion, use_custom):
    myuuid = uuid.uuid4()
    s3_file_path_music = f"sound/{str(myuuid)}.wav"

    model = musicgen.MusicGen.get_pretrained(model_name, device="cuda")
    if use_custom:
        model.lm.load_state_dict(torch.load('models/lm_final.pt'))
    model.set_generation_params(
        duration=duration,
        use_sampling=sampling,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    wav = model.generate([prompt], progress=True, return_tokens=True)
    if use_diffusion:
        mbd = MultiBandDiffusion.get_mbd_musicgen()
        diff = mbd.tokens_to_wav(wav[1])
        audio_write(diff, str(myuuid), model.sample_rate, strategy='loudness')
    else:
        audio_write(wav[0], str(myuuid), model.sample_rate, strategy='loudness')

    with open(f"{str(myuuid)}.wav", "rb") as f:
        wav_data = f.read()
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        s3.put_object(Bucket=BUCKET_NAME, Body=wav_data, Key=s3_file_path_music, ContentType="audio/wav")

    return s3_file_path_music

# 이미지를 생성하고 S3에 업로드하는 함수
def generate_and_upload_image(image_prompt):
    image_stream = generate_image(image_prompt)
    image_file_name = f"{uuid.uuid4()}_album_cover.png"
    s3_file_path_image = f"image/{image_file_name}"

    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    s3.put_object(Bucket=BUCKET_NAME, Body=image_stream.read(), Key=s3_file_path_image, ContentType="image/png")

    return s3_file_path_image





if __name__ == "__main__":
    app.run('0.0.0.0', port=7777, debug=True)
