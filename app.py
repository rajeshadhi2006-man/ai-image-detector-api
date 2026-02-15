import os

# Force CPU usage to avoid GPU errors/hangs on machines without proper CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging clutter

import io
import time
import sys
import logging
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ai-detector-api")

logger.info("Initializing FastAPI app...")

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    logger.info("FastAPI libraries imported.")
except Exception as e:
    logger.error(f"Error importing FastAPI libraries: {e}")
    sys.exit(1)

logger.info("Importing TensorFlow...")
try:
    with io.StringIO() as f:
        # Redirect stdout to suppress TF import messages if necessary, 
        # but here we just want to catch the error
        import tensorflow as tf
    logger.info(f"TensorFlow {tf.__version__} imported successfully.")
except Exception as e:
    logger.error(f"Error importing TensorFlow: {e}")


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
        logger.info(f"Loading model from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Using dummy prediction logic.")
        return None

# Load the model once at startup
logger.info("Starting model load...")
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
            logger.error(f"Prediction error: {e}")
            return False, 0.0
    else:
        # Fallback dummy logic if model isn't loaded
        logger.warning("Using fallback logic (No model loaded)")
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
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
