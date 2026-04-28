from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_model, predict

app = FastAPI(title="Indian Food Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load_model()

@app.get("/")
def root():
    return {"message": "Indian Food Classifier API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(model, image_bytes)
    return result