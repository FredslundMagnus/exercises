from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
import re
from pydantic import BaseModel
import cv2

app = FastAPI()

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
def predict_step(image_paths: list[str]):
   images = []
   for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
         i_image = i_image.convert(mode="RGB")

      images.append(i_image)
   print(image_paths)
   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   print(pixel_values)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   print(output_ids)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   print(preds)
   return preds

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = 28, w: int = 28):
   with open('image.jpg', 'wb') as image:
      content = await data.read()
      image.write(content)
      image.close()
   # img = cv2.imread("image.jpg")
   # res = cv2.resize(img, (h, w))
   
   # cv2.imwrite('image_resize.jpg', res)
   
   response = {
      "input": data,
      "data": predict_step(['image.jpg'])[0],
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
   }
   
   return response