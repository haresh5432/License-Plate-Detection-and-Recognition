import cv2
import imutils
import os
import sys
import torch
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import deque, Counter

def load_recognition_models():
    """
    Initializes and loads the plate detection (YOLO) and character
    recognition (EasyOCR) models, optimizing for Apple Silicon GPU.
    """
   

    if torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon GPU (MPS) detected. Using GPU for acceleration.")
    else:
        device = 'cpu'
        print("Apple Silicon GPU not found. Using CPU.")


    model_path = "/Users/hareshshokeen/Desktop/CV/runs/detect/number_plate_quick/weights/best.pt"
    if not os.path.isfile(model_path):
        print(f"YOLO model not found at path: {model_path}")
        sys.exit(1)

    print("Loading AI models, please wait...")
    try:

        # Load the YOLO model onto the detected device (GPU or CPU)

        plate_detector = YOLO(model_path).to(device)
        
        # Initialize EasyOCR and tell it to use the GPU if available

        text_reader = easyocr.Reader(['en'], gpu=(device == 'mps'))
        
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        sys.exit(1)
        
    print("Models loaded successfully.")
    return plate_detector, text_reader, device

def analyze_frame_for_license_plate(frame, plate_detector, text_reader, device, recent_detections=None):
    """
    Analyzes a single image frame to detect and read a license plate,
    stabilizing the text output if a history of detections is provided.
    """
    display_frame = frame.copy()
    
    # Run detection, ensuring the model uses the correct device

    detection_results = plate_detector(frame, device=device, verbose=False)

    current_frame_texts = []
    for result in detection_results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            ocr_output = text_reader.readtext(plate_crop)
            if ocr_output:
                raw_text = "".join(res[1] for res in ocr_output).upper().replace(" ", "")
                if len(raw_text) > 4 and any(char.isdigit() for char in raw_text):
                    current_frame_texts.append(raw_text)

    if recent_detections is not None and current_frame_texts:
        recent_detections.append(current_frame_texts[0])

    final_text_to_display = ""
    if recent_detections is not None and len(recent_detections) > 0:
        final_text_to_display = Counter(recent_detections).most_common(1)[0][0]
        print(f"Stabilized Reading: {final_text_to_display}")
    elif current_frame_texts:
        final_text_to_display = current_frame_texts[0]
        print(f"Detected Plate: {final_text_to_display}")

    for result in detection_results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, final_text_to_display, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return display_frame

def run_application():
    """
    Manages the main user interface, input selection, and application loop.
    """
    plate_detector, text_reader, device = load_recognition_models()
    media_source = None
    is_paused = False
    recent_detections = deque(maxlen=20)

    while True:
        if media_source is None:
            print("\n--- License Plate Reader Menu ---")
            print("1: Analyze an image file")
            print("2: Analyze a video file")
            print("3: Use live webcam")
            print("4: Use phone camera")
            print("q: Quit")
            choice = input("Select an option: ").strip().lower()

            if choice == '1':
                path = input("Enter the path to your image: ").strip()
                if not os.path.isfile(path):
                    print("That file does not exist. Please try again.")
                    continue
                frame = cv2.imread(path)
                if frame is None:
                    print("Could not read the image file.")
                    continue
                processed_frame = analyze_frame_for_license_plate(frame, plate_detector, text_reader, device)
                cv2.imshow("Image Analysis (Press any key to close)", processed_frame)
                cv2.waitKey(0)
                cv2.destroyWindow("Image Analysis")

            elif choice == '2':
                path = input("Enter the path to your video: ").strip()
                if not os.path.isfile(path):
                    print("That file does not exist. Please try again.")
                    continue
                media_source = cv2.VideoCapture(path)
                recent_detections.clear()
                if not media_source.isOpened():
                    print("Could not open the video file.")
                    media_source = None

            elif choice == '3':
                media_source = cv2.VideoCapture(0)
                recent_detections.clear()
                if not media_source.isOpened():
                    print("Could not access the webcam.")
                    media_source = None

            elif choice == '4':
                url = input("Enter the IP Webcam URL from your phone (e.g., http://192.168.1.5:8080): ").strip()
                video_url = f"{url}/video"
                media_source = cv2.VideoCapture(video_url)
                recent_detections.clear()
                if not media_source.isOpened():
                    print("Could not connect to the phone's camera stream. Check the URL and Wi-Fi connection.")
                    media_source = None
            
            elif choice == 'q':
                break
            
            else:
                print("Invalid selection. Please choose a valid option.")

        if media_source is not None:
            if not is_paused:
                was_read, frame = media_source.read()
                if not was_read:
                    print("Finished video or stream lost. Returning to menu.")
                    media_source.release()
                    media_source = None
                    cv2.destroyWindow("Live Analysis")
                    continue
                
                frame = imutils.resize(frame, width=800)
                processed_frame = analyze_frame_for_license_plate(frame, plate_detector, text_reader, device, recent_detections)
                cv2.imshow("Live Analysis (p=pause, m=menu, q=quit)", processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                media_source.release()
                media_source = None
                cv2.destroyWindow("Live Analysis")
            elif key == ord('p'):
                is_paused = not is_paused
                print("Paused" if is_paused else "Resumed")
            elif key == ord('q'):
                break

    print("Closing application.")
    if media_source:
        media_source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_application()
