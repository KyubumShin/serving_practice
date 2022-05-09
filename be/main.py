import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, UploadFile, File
from torchvision import models
from PIL import Image
from io import BytesIO

app = FastAPI()


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])


@app.get("/")
async def root():
    return {"message": "Hello World"}


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.post("/csf")
async def classfication(file: UploadFile = File(...)):
    image = read_imagefile(await file.read()).convert("RGB")
    model = models.resnet34(pretrained=True)
    input_image = transform(image)
    pred = model(input_image.unsqueeze(0))
    output = pred.argmax(dim=1)
    return {'result': output.item()}


if __name__ == "__main__":
    uvicorn.run(app, reload=True)
