import numpy as np

def score_position(saliency, x, y, w, h):
    region = saliency[y:y+h, x:x+w]

    if region.size == 0:
        return -999999

    return -np.mean(region)  # lower = better


def choose_best_position(image, saliency, logo_shape):
    h, w = image.shape[:2]
    lh, lw = logo_shape[:2]

    margin = 30

    positions = [
        (margin, margin),
        (w - lw - margin, margin),
        (margin, h - lh - margin),
        (w - lw - margin, h - lh - margin)
    ]

    best_score = -999999
    best_pos = positions[0]

    for (x, y) in positions:
        score = score_position(saliency, x, y, lw, lh)
        if score > best_score:
            best_score = score
            best_pos = (x, y)

    return best_pos


def choose_logo_variant(brightness):
    if brightness > 140:   # improved threshold
        return "dark"
    else:
        return "light"