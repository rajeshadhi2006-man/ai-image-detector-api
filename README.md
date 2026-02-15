# AI Image Detector API

This is a Flask-based API for detecting AI-generated images.

## Project Structure

- `app.py`: Main application file containing the API endpoints.
- `model/`: Directory to store the trained ML model (e.g., `.keras` or `.h5` files).
- `uploads/`: Temporary storage for uploaded images (if needed).
- `requirements.txt`: Python dependencies.

## Setup

1.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```

The API will be available at `http://localhost:5000`.

## API Endpoints

### 1. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "AI Image Detector API"
  }
  ```

### 2. Predict Image
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: `multipart/form-data` with a key `file` containing the image.
- **Response**:
  ```json
  {
    "filename": "image.jpg",
    "is_ai_generated": true,
    "confidence": 0.92,
    "prediction_label": "AI-Generated",
    "timestamp": 1707923456.789
  }
  ```

## Model Integration

Currently, the `app.py` uses a placeholder logic in `load_model()` and `predict_is_ai_generated()`. To integrate a real model:

1.  Place your model file in the `model/` directory.
2.  Update `MODEL_PATH` in `app.py`.
3.  Uncomment the model loading and prediction code in `app.py`.
