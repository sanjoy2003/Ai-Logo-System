import cv2
import numpy as np

def overlay_logo(image, logo, x, y, opacity=1.0):
    lh, lw = logo.shape[:2]

    if y + lh > image.shape[0] or x + lw > image.shape[1]:
        return image

    if logo.shape[2] == 4:
        alpha = logo[:, :, 3] / 255.0
        logo_rgb = logo[:, :, :3]
    else:
        alpha = np.ones((lh, lw))
        logo_rgb = logo

    # 🔥 NO blur here → keep sharp
    alpha = alpha * opacity

    # 🔥 HARD CLEAN BLEND
    for c in range(3):
        image[y:y+lh, x:x+lw, c] = (
            alpha * logo_rgb[:, :, c] +
            (1 - alpha) * image[y:y+lh, x:x+lw, c]
        )

    return image