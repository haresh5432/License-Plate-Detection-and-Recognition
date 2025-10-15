# segmentation/segment.py
import cv2
import numpy as np

def region_otsu_binarize(img, grid=(3, 3)):
    """Divide plate into sub-regions and apply Otsu threshold locally."""
    h, w = img.shape[:2]
    bin_img = np.zeros((h, w), dtype=np.uint8)
    gh, gw = grid
    for i in range(gh):
        for j in range(gw):
            y0, y1 = int(i * h / gh), int((i + 1) * h / gh)
            x0, x1 = int(j * w / gw), int((j + 1) * w / gw)
            cell = img[y0:y1, x0:x1]
            gray = cell if len(cell.shape) == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(thresh) > 127:
                thresh = cv2.bitwise_not(thresh)
            bin_img[y0:y1, x0:x1] = thresh
    return bin_img

def estimate_slant_angle(bin_img):
    """Estimate slant angle using symmetric horizontal projections."""
    h, w = bin_img.shape
    left_half = bin_img[:, :w//2]
    right_half = bin_img[:, w//2:]
    proj_left = np.sum(left_half > 0, axis=1)
    proj_right = np.sum(right_half > 0, axis=1)
    def centroid(proj):
        total = np.sum(proj)
        return int(np.sum(np.arange(len(proj)) * proj) / (total + 1e-6))
    cl, cr = centroid(proj_left), centroid(proj_right)
    H = abs(cl - cr)
    W = np.sum(np.sum(bin_img, axis=0) == 0)
    theta = np.degrees(np.arctan2(H, W + 1e-6))
    return theta if cr > cl else -theta

def deskew_plate(plate_img, angle):
    """Rotate plate to correct slant."""
    (h, w) = plate_img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def algorithm1_split(boxes, ref_w, gamma1=1.5, gamma2=2.5):
    """Algorithm 1 from paper: split wide components."""
    S = []
    for (x, y, w, h, area) in boxes:
        if w <= ref_w * gamma1:
            S.append((x, y, w, h))
        elif w <= ref_w * gamma2:
            S.append((x, y, w//2, h))
            S.append((x + w//2, y, w - w//2, h))
        else:
            n = int(np.ceil(w / (ref_w * gamma1)))
            split_w = w // n
            for k in range(n):
                sx = x + k * split_w
                sw = split_w if k < n - 1 else w - split_w * (n - 1)
                S.append((sx, y, sw, h))
    return sorted(S, key=lambda b: b[0])

def segment_plate(plate_img):
    """Main segmentation pipeline."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    bin_mask = region_otsu_binarize(gray)
    angle = estimate_slant_angle(bin_mask)
    plate_rot = deskew_plate(plate_img, angle)
    gray_rot = cv2.cvtColor(plate_rot, cv2.COLOR_BGR2GRAY)
    _, bin_final = cv2.threshold(gray_rot, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_final) > 127:
        bin_final = cv2.bitwise_not(bin_final)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_final, connectivity=8)
    boxes = [(x, y, w, h, area) for i, (x, y, w, h, area) in enumerate(stats[1:])]
    ref_w = max([b[2] for b in boxes]) / 2
    segments = algorithm1_split(boxes, ref_w)
    chars = []
    for (x, y, w, h) in segments:
        crop = gray_rot[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (32, 32))
        chars.append(resized)
    return chars
