from fastapi import FastAPI
from http import HTTPStatus
import re
from pydantic import BaseModel
import cv2
from fastapi.responses import FileResponse


from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}
def predict_step(image_paths):
   images = []
   for image_path in image_paths:
      i_image = Image.open(image_path)
      if i_image.mode != "RGB":
         i_image = i_image.convert(mode="RGB")

      images.append(i_image)
   pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
   pixel_values = pixel_values.to(device)
   output_ids = model.generate(pixel_values, **gen_kwargs)
   preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
   preds = [pred.strip() for pred in preds]
   return preds



app = FastAPI()


@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/items/{item_id}")
def read_item(item_id: int):
   return {"item_id": item_id}

from enum import Enum
class ItemEnum(Enum):
   alexnet = "alexnet"
   resnet = "resnet"
   lenet = "lenet"

@app.get("/restric_items/{item_id}")
def read_restricted_item(item_id: ItemEnum):
   return {"item_id": item_id}

@app.get("/query_items")
def query_item(item_id: int):
   return {"item_id": item_id}

database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
   username_db = database['username']
   password_db = database['password']
   if username not in username_db and password not in password_db:
      with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
      username_db.append(username)
      password_db.append(password)
   return "login saved"

class Item(BaseModel):
   email: str
   domain_match: str

@app.get("/text_model/")
def contains_email(data: Item):
   regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
   response = {
      "input": data,
      "message": HTTPStatus.OK.phrase,
      "status-code": HTTPStatus.OK,
      "is_email": re.fullmatch(regex, data.email) is not None and data.domain_match in data.email
   }
   return response

from fastapi import UploadFile, File

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = 28, w: int = 28):
   with open('image.jpg', 'wb') as image:
      content = await data.read()
      image.write(content)
      image.close()
   img = cv2.imread("image.jpg")
   res = cv2.resize(img, (h, w))
   
   cv2.imwrite('image_resize.jpg', res)
   
   return FileResponse('image_resize.jpg')
   # response = {
   #    "input": data,
   #    "file": FileResponse('image_resize.jpg'),
   #    "message": HTTPStatus.OK.phrase,
   #    "status-code": HTTPStatus.OK,
   # }
   # return response

