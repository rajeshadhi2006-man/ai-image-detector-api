import os

# Force CPU usage to avoid GPU errors/hangs on machines without proper CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging clutter

import io
import time
import sys
import numpy as np
from PIL import Image

print("Initializing FastAPI app...", flush=True)

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    print("FastAPI libraries imported.", flush=True)
except Exception as e:
    print(f"Error importing FastAPI libraries: {e}", flush=True)
    sys.exit(1)

print("Importing TensorFlow...", flush=True)
try:
    import tensorflow as tf
    print("TensorFlow imported successfully.", flush=True)
except Exception as e:
    print(f"Error importing TensorFlow: {e}", flush=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'model/ai_image_detector_final.keras'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="AI Image Detector API",
    description="API for detecting AI-generated images using TensorFlow/Keras",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    """
    Load the trained Keras model.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...", flush=True)
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!", flush=True)
            return model
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            return None
    else:
        print(f"Model file not found at {MODEL_PATH}. Using dummy prediction logic.", flush=True)
        return None

# Load the model once at startup
print("Starting model load...", flush=True)
MODEL = load_model()

def preprocess_image(image: Image.Image):
    """
    Preprocess the PIL Image for your model.
    """
    # Resize to the input size your model expects (e.g., 224x224)
    image = image.resize((224, 224)) 
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array / 255.0  
    
    # Add batch dimension (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_is_ai_generated(image: Image.Image):
    """
    Perform the prediction.
    Returns: Is AI Generated (bool), Confidence (float)
    """
    if MODEL:
        try:
            processed_img = preprocess_image(image)
            prediction = MODEL.predict(processed_img)
            
            # Assuming the model outputs a single probability score (0 to 1)
            score = float(prediction[0][0]) 
            
            is_ai = score > 0.5
            return is_ai, score
        except Exception as e:
            print(f"Prediction error: {e}", flush=True)
            return False, 0.0
    else:
        # Fallback dummy logic if model isn't loaded
        print("Using fallback logic (No model loaded)", flush=True)
        width, height = image.size
        return (width == 1024), 0.5

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "AI Image Detector API is running (FastAPI). Use /predict to analyze images.",
        "docs_url": "/docs" 
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI Image Detector API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file extension
    filename = file.filename
    if not filename or '.' not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}")

    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Predict
        is_ai, confidence = predict_is_ai_generated(image)
        
        return {
            "filename": filename,
            "is_ai_generated": bool(is_ai),
            "confidence": float(confidence),
            "prediction_label": "AI-Generated" if is_ai else "Real",
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
