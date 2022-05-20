from io import BytesIO
import pandas as pd
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Response

from model import get_model, predict_image

app = FastAPI()

model = get_model()



def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file)).convert("RGB")
    return image


@app.post('/seg')
async def predict(file: UploadFile = File()):
    image = read_image_file(await file.read())
    mask = predict_image(model, image)
    im = Image.fromarray(mask)
    with BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="mask.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')

