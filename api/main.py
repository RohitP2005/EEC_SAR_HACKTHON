from fastapi import FastAPI, UploadFile, File,HTTPException
from fastapi.responses import StreamingResponse,JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import torch
import numpy as np
from PIL import Image
import io
from skimage.color import rgb2lab
from model import MainModel
from utils import lab_to_rgb
from io import BytesIO
from pydantic import BaseModel
import pyproj
import cv2
import datetime
from requests_toolbelt.multipart import decoder

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = MainModel()
model.load_state_dict(torch.load("./best_model.pth", map_location=device))
model.to(device)
model.eval()

# --- Sentinel Hub Credentials ---
CLIENT_ID = "60c275fa-6351-4cad-866a-693780befc30"
CLIENT_SECRET = "BdcyiEAzBLkWgwEx1vyfONJcIamsQ3JW"
AUTH_URL = "https://services.sentinel-hub.com/oauth/token"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"

# ------------------------------------------------------------------------------------
# Colorisatiion Inference 
# ------------------------------------------------------------------------------------
def preprocess_image(image):
    # Load image with PIL
    # Ensure it's 3-channel grayscale (for rgb2lab)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    # Convert RGB to Lab
    image_lab = rgb2lab(image).astype("float32")
    L_np = image_lab[:, :, 0]
    L_norm = (L_np / 50.0) - 1.0  # Normalize to [-1, 1]

    # Convert to tensor
    L_tensor = torch.tensor(L_norm).unsqueeze(0).unsqueeze(0).to(device)
    return L_tensor, L_np


def postprocess_and_return_image(L_tensor, ab_pred):
    rgb_pred = lab_to_rgb(L_tensor.cpu(), ab_pred)  # Shape: [1, H, W, 3]
    rgb_image = (rgb_pred[0] * 255).astype(np.uint8)  # Convert to uint8

    # Convert NumPy array to PIL Image
    image = Image.fromarray(rgb_pred[0])

    # Save to BytesIO buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer



@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert PIL Image to NumPy array
    image_array = np.array(image)

    # Preprocess
    L_tensor, L_np = preprocess_image(image_array)

    # Inference
    with torch.no_grad():
        ab_pred = model.net_G(L_tensor).cpu()

    # Get final image buffer
    image_buffer = postprocess_and_return_image(L_tensor, ab_pred)

    return StreamingResponse(image_buffer, media_type="image/png")


# ------------------------------------------------------------------------------------
# Sentinal Requets 
# ------------------------------------------------------------------------------------

# Pydantic request model
class Coordinates(BaseModel):
    latitude: float
    longitude: float
    
def get_access_token():
    print("[INFO] Requesting access token...")
    response = requests.post(
        AUTH_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    print("[INFO] Access token obtained.")
    return token

def convert_bbox_epsg4326_to_3857(bbox):
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return list(transformer.transform(bbox[0], bbox[1])) + list(transformer.transform(bbox[2], bbox[3]))

def process_sar_image(input_path, output_path):
    try:
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image")
        enhanced = cv2.equalizeHist(img)
        denoised = cv2.medianBlur(enhanced, 3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        final_image = clahe.apply(denoised)
        cv2.imwrite(output_path, final_image)
        return True
    except Exception as e:
        print(f"[ERROR] Image processing failed: {e}")
        return False
    
def preprocess_for_model(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image.")
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)  # (1, 256, 256)
    img = np.expand_dims(img, axis=0)  # (1, 1, 256, 256)
    return img


@app.post("/get-sar-image")
async def get_sar_image(coords: Coordinates):
    lat, lon = coords.latitude, coords.longitude
    print(lat,lon)
    try:
        token = get_access_token()
    except Exception as e:
        print("Token Error")
        raise HTTPException(status_code=500, detail=f"Failed to get token: {e}")

    margin = 0.05
    bbox = [lon - margin, lat - margin, lon + margin, lat + margin]
    bbox_3857 = convert_bbox_epsg4326_to_3857(bbox)

    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    past = now - datetime.timedelta(days=365)
    print(now,past)
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["VH"],
            output: { bands: 1 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.VH];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox_3857,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"}
            },
            "data": [{
                "type": "S1GRD",
                "dataFilter": {
                    "timeRange": {
                        "from": past.isoformat(),
                        "to": now.isoformat()
                    }
                }
            }]
        },
        "evalscript": evalscript,
        "responses": [
            {"identifier": "default", "format": {"type": "image/png"}},
            {"identifier": "userdata", "format": {"type": "application/json"}}
        ],
        "output": {"width": 256, "height": 256}
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    print("[INFO] Requesting SAR image from Sentinel Hub...")
    try:
        response = requests.post(PROCESS_URL, headers=headers, json=payload)
        print(response)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise HTTPException(status_code=500, detail=f"HTTP Error: {err} - {response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e}")

    image_path = "latest_sar_image.png"
    processed_image_path = "processed_sar_image.png"
    metadata = {}

    content_type = response.headers.get("Content-Type", "")
    if "multipart" in content_type:
        multipart_data = decoder.MultipartDecoder.from_response(response)
        for part in multipart_data.parts:
            ctype = part.headers.get(b"Content-Type", b"").decode()
            if "image" in ctype:
                with open(image_path, "wb") as f:
                    f.write(part.content)
            elif "application/json" in ctype:
                metadata = part.content.decode()
    elif "image/png" in content_type:
        with open(image_path, "wb") as f:
            f.write(response.content)

    if process_sar_image(image_path, processed_image_path):
        try:
            model_input = preprocess_for_model(processed_image_path)
            return JSONResponse(content={
                "message": "SAR image processed and ready for model",
                "shape": model_input.shape,
                "image_url": "http://localhost:8000/image",
                "metadata": metadata
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")
    else:
        raise HTTPException(status_code=500, detail="Failed to process SAR image.")

@app.get("/image")
async def serve_image():
    return FileResponse("processed_sar_image.png", media_type="image/png")