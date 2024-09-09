from ultralytics import YOLO
import cv2
from sort.sort import Sort
from utils import get_car, license_plate_reader, store_readings_to_mysql
import numpy as np
import tensorflow as tf
import mysql.connector


# Establish MySQL connection
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user='root',
        password='',
        database='rohan'
    )
    mycursor = mydb.cursor()
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit(1)

def remove_ind_from_license_plate(license_plate_text):
    if "IND" in license_plate_text:
        license_plate_text = license_plate_text.replace("IND", "").strip()
        license_plate_text = ''.join(license_plate_text.split())
        return license_plate_text
    else:
        license_plate_text = ''.join(license_plate_text.split())
        return license_plate_text.strip()


MODEL_PATH = "D:/Internships/Intel_Unnati/Final/backend/content_rest/self_model_best.h5"
model = tf.keras.models.load_model(MODEL_PATH)

results = {}
mot_tracker = Sort()

coco_model = YOLO('yolov8l.pt')
license_plate_detector = YOLO('D:/Internships/Intel_Unnati/Final/backend/model1/best.pt')
vehicles = [2, 3, 5, 7]

# Read the frame
frame = cv2.imread('backend/Images_check/t4.jpeg')
frame_nmr = 1
results = {frame_nmr: {}}

# Detect vehicles
detections = coco_model(frame)[0]
vehicle_detections = [det[:5] for det in detections.boxes.data.tolist() if int(det[5]) in vehicles]

# Track vehicles
track_ids = mot_tracker.update(np.array(vehicle_detections))

# Set license plate detection threshold
license_plate_threshold = 0.7

# Detect license plates
license_plates = license_plate_detector(frame)[0]

for license_plate in license_plates.boxes.data.tolist():
    score = license_plate[4]
    if score < license_plate_threshold:
        continue
    
    x1, y1, x2, y2, _, class_id = license_plate
    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
    print(car_id)
    
    # Crop and process the license plate
    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.imshow('original_crop', license_plate_crop)
    cv2.waitKey(0)
    
    # Read license plate text using ResNet-50 model
    license_plate_text_resnet, license_plate_text_easyocr, license_plate_text_tes, license_plate_text_score = license_plate_reader(license_plate_crop, model, MODEL_PATH)
    
    license_plate_text_easyocr_c = remove_ind_from_license_plate(license_plate_text_easyocr)
    license_plate_text_tes_c = remove_ind_from_license_plate(license_plate_text_easyocr)

    # Print results for comparison
    print("License Plate Text (ResNet-50):", license_plate_text_resnet)
    print("License Plate Text (EasyOCR):", license_plate_text_easyocr_c)
    print("License Plate Text (TesseractOCR):", license_plate_text_tes_c)

    if license_plate_text_resnet is not None:
        results[frame_nmr][car_id] = {
            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'text_resnet': license_plate_text_resnet,
                'text_easyocr': license_plate_text_easyocr_c,
                'text_tes': license_plate_text_tes_c,
                'text_score': license_plate_text_score
            }
        }


# Store readings to MySQL database
_, status = store_readings_to_mysql(results, mycursor, mydb)
print(status)

# Close MySQL connection
mydb.close()
