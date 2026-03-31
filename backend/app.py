from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import os
import uuid

from core.vision import get_brightness_map, get_saliency_map
from core.decision import choose_best_position, choose_logo_variant
from core.render import overlay_logo

app = FastAPI()

# Logo paths (relative to backend folder)
LOGO_LIGHT = "assets/logo_light.png"
LOGO_DARK = "assets/logo_dark.png"


# Root check
@app.get("/")
def home():
    return {"message": "Logo AI System Running"}


# Main API
@app.post("/apply-logo/")
async def apply_logo(file: UploadFile = File(...)):
    try:
        # =========================
        # 1. Read and decode image
        # =========================
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            return {"error": "Invalid or corrupted image"}

        # =========================
        # 2. Analyze image
        # =========================
        brightness = get_brightness_map(image)

        saliency = get_saliency_map(image)
        if saliency is None:
            saliency = np.zeros(image.shape[:2], dtype=np.uint8)

        # =========================
        # 3. Choose logo variant
        # =========================
        variant = choose_logo_variant(brightness)
        logo_path = LOGO_DARK if variant == "dark" else LOGO_LIGHT

        # =========================
        # 4. Load logo
        # =========================
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

        if logo is None:
            return {"error": f"Logo not found at {logo_path}"}
        # enhance logo contrast
        logo_rgb = logo[:, :, :3]

        logo_rgb = cv2.convertScaleAbs(logo_rgb, alpha=1.2, beta=10)

        if logo.shape[2] == 4:
            logo[:, :, :3] = logo_rgb
        # =========================
        # 5. Resize logo
        # =========================
        h, w = image.shape[:2]

        logo_h, logo_w = logo.shape[:2]

        target_w = int(w * 0.12)
        ratio = target_w / logo_w
        target_h = int(logo_h * ratio)

        logo = cv2.resize(
            logo,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4
        )
        # =========================
        # 6. Decide position
        # =========================
        x, y = choose_best_position(image, saliency, logo.shape)

        # =========================
        # 7. Apply logo
        # =========================
        output = overlay_logo(image, logo, x, y, opacity=0.8)

        if output is None:
            return {"error": "Overlay failed"}

        # =========================
        # 8. Save output
        # =========================
        os.makedirs("outputs", exist_ok=True)

        output_filename = f"output_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join("outputs", output_filename)

        success = cv2.imwrite(output_path, output)

        if not success:
            return {"error": "Failed to save output image"}

        # =========================
        # 9. Response
        # =========================
        return {
            "message": "Logo applied successfully",
            "file": output_filename,
            "path": output_path
        }

    except Exception as e:
        return {"error": str(e)}