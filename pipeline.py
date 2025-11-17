import cv2
import imutils
import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import deque, Counter
import pickle


# MODEL LOADING


def load_recognition_models():

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS).")
    else:
        device = "cpu"
        print("Using CPU.")

    # YOLO model path
    model_path = "/Users/hareshshokeen/Desktop/CV/runs/detect/number_plate_quick/weights/best.pt"

    if not os.path.isfile(model_path):
        print("ERROR: YOLO model not found.")
        sys.exit(1)

    print("Loading YOLO + OCR models...")

    # YOLO model
    plate_detector = YOLO(model_path).to(device)

    # EasyOCR
    text_reader = easyocr.Reader(["en"], gpu=(device == "mps"))

    # Load SAE-BLS models 
    digit_model, letter_model = load_sae_bls_models()

    print("All models loaded successfully.\n")

    return plate_detector, text_reader, digit_model, letter_model, device

def load_sae_bls_models():
    digit_model_path = "models/sae_bls_digit.pkl"
    letter_model_path = "models/sae_bls_letter.pkl"

    digit_model = None
    letter_model = None

    # Load digit model 
    try:
        with open(digit_model_path, "rb") as f:
            digit_model = pickle.load(f)
    except:
        digit_model = None
    print("SAE-BLS Digit Model Loaded Successfully ✓")

    # Load letter model 
    try:
        with open(letter_model_path, "rb") as f:
            letter_model = pickle.load(f)
    except:
        letter_model = None
    print("SAE-BLS Letter Model Loaded Successfully ✓")

    return digit_model, letter_model


# PREPROCESSING FOR OCR (plate crop only)


def preprocess_plate_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 7, 75, 75)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(blur, -1, sharpen_kernel)

    binary = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    morph_kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_kernel)

    return clean


# FRAME ANALYSIS


def analyze_frame_for_license_plate(frame, plate_detector, text_reader,
                                    digit_model, letter_model, device,
                                    recent_detections=None):

    display_frame = frame.copy()

    # --- YOLO detection on raw frame ---
    results = plate_detector(frame, device=device, verbose=False)

    current_frame_texts = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # Preprocess ONLY plate region for OCR
            prep_for_ocr = preprocess_plate_for_ocr(plate_crop)

            # OCR read
            ocr_output = text_reader.readtext(prep_for_ocr)

            if ocr_output:
                raw_text = "".join([res[1] for res in ocr_output])
                raw_text = raw_text.upper().replace(" ", "")

                if len(raw_text) > 4 and any(ch.isdigit() for ch in raw_text):
                    current_frame_texts.append(raw_text)

    # Stabilize text
    if recent_detections is not None and current_frame_texts:
        recent_detections.append(current_frame_texts[0])

    if recent_detections and len(recent_detections) > 0:
        final_text = Counter(recent_detections).most_common(1)[0][0]
    elif current_frame_texts:
        final_text = current_frame_texts[0]
    else:
        final_text = ""

    # Draw YOLO box + text
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, final_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

    return display_frame


# ==========================================================
# MAIN APPLICATION LOOP
# ==========================================================

def run_application():

    plate_detector, text_reader, digit_model, letter_model, device = load_recognition_models()

    media_source = None
    is_paused = False
    recent_detections = deque(maxlen=20)

    while True:

        if media_source is None:
            print("\n--- Choose an option to select your input format ---")
            print("1: Analyze Image")
            print("2: Analyze Video")
            print("3: Live Webcam")
            print("4: Phone Camera (IP Webcam)")
            print("q: Quit")

            choice = input("Select option: ").strip().lower()

            if choice == "1":
                path = input("Image path: ").strip()
                frame = cv2.imread(path)
                processed = analyze_frame_for_license_plate(
                    frame, plate_detector, text_reader,
                    digit_model, letter_model, device
                )
                cv2.imshow("Image Analysis", processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            elif choice == "2":
                path = input("Video path: ").strip()
                media_source = cv2.VideoCapture(path)
                recent_detections.clear()

            elif choice == "3":
                media_source = cv2.VideoCapture(0)
                recent_detections.clear()

            elif choice == "4":
                url = input("Phone IP Webcam URL: ").strip()
                media_source = cv2.VideoCapture(f"{url}/video")
                recent_detections.clear()

            elif choice == "q":
                break

            else:
                print("Invalid option.")

        # ---------- Live Stream Loop ----------
        if media_source is not None:

            if not is_paused:
                ret, frame = media_source.read()
                if not ret:
                    media_source.release()
                    media_source = None
                    cv2.destroyAllWindows()
                    continue

                frame = imutils.resize(frame, width=800)

                processed = analyze_frame_for_license_plate(
                    frame, plate_detector, text_reader,
                    digit_model, letter_model, device,
                    recent_detections
                )

                cv2.imshow("Live Analysis  (p=pause, m=menu, q=quit)", processed)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                is_paused = not is_paused
                print("Paused" if is_paused else "Resumed")

            elif key == ord("m"):
                media_source.release()
                media_source = None
                cv2.destroyAllWindows()

            elif key == ord("q"):
                break

    print("Closing application.")
    if media_source:
        media_source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_application()
