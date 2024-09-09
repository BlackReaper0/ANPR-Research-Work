import string
import easyocr
import cv2
import numpy as np
import tensorflow as tf
import datetime
from datetime import date, datetime
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import hashlib
import time
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'T':'1'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def correct_license_plate(ocr_result):
    # Expected format: AA99AA9999
    corrected_plate = []

    for i, char in enumerate(ocr_result):
        if i < 2 or (i >= 4 and i < 6):  # Expecting alpha characters
            if char in dict_int_to_char:
                corrected_plate.append(dict_int_to_char[char])
            else:
                corrected_plate.append(char)
        else:  # Expecting numeric characters
            if char in dict_char_to_int:
                corrected_plate.append(dict_char_to_int[char])
            else:
                corrected_plate.append(char)

    return ''.join(corrected_plate)

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for track_id in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track_id
        
        # Convert all coordinates to floats
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        xcar1, ycar1, xcar2, ycar2 = float(xcar1), float(ycar1), float(xcar2), float(ycar2)
        
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1

def img_op(image):
    image = Image.fromarray(image)

    gray_image = image.convert("L")

    # Increase contrast
    contrast_enhancer = ImageEnhance.Contrast(gray_image)
    high_contrast_image = contrast_enhancer.enhance(3.0)

    # Sharpen the image
    sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
    sharper_image_for_ocr = sharpness_enhancer.enhance(2.0)
    # Convert the PIL image to a NumPy array
    sharper_image_for_ocr_np = np.array(sharper_image_for_ocr)

    return sharper_image_for_ocr

def predict(image, model):
    class_names = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)
    pred = np.argmax(result, axis=-1)[0]
    return class_names[pred], pred

def license_plate_reader(image, model, weights):
    total_prob = 0.0
    num_chars = 0
    args = {"image": image, "model": weights}

    # Define the new dimensions (width, height)
    new_width = 1000
    new_height = 350

    image_org=image

    # Resize the image
    image = cv2.resize(image, (new_width, new_height))
    cv2.imshow('plate', image)
    cv2.waitKey(0)

    image_ocr = img_op(image_org)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_width = 11  # Adjust this value to change the width of the kernel
    kernel_height = 11  # Adjust this value to change the height of the kernel
    std_dev_x = 0  # Adjust this value to control blurring along the x-axis
    std_dev_y = 0  # Adjust this value to control blurring along the y-axis

    blurred = cv2.GaussianBlur(gray, (kernel_width, kernel_height), std_dev_x, std_dev_y)

    thresh = cv2.adaptiveThreshold(blurred, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    
    # Perform connected components analysis on the thresholded images and
    # initialize the mask to hold only the components we are interested in
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Set lower bound and upper bound criteria for characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 240  # heuristic param, can be fine tuned if necessary
    upper = total_pixels // 50  # heuristic param, can be fine tuned if necessary

    # Loop over the unique components
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    # Find contours and get bounding box for each contour
    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # Sort the bounding boxes from left to right, top to bottom
    # sort by Y first, and then sort by X if Ys are similar

    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[0] - rect2[0]
        else:
            return rect1[1] - rect2[1]
    
    # Sort the bounding boxes from left to right
    boundingBoxes = sorted(boundingBoxes, key=lambda x: x[0])
    
    # Define constants
    TARGET_WIDTH = 128
    TARGET_HEIGHT = 128

    chars = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    vehicle_plate = ""


    
    # Loop over the bounding boxes
    for rect in boundingBoxes:

        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = mask[y:y+h, x:x+w]
        crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT -
                    rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (
            TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY,
                                paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        char, prob = predict(crop,model)
        total_prob += np.max(prob)
        num_chars += 1

        vehicle_plate += char
        idx = np.argmax(prob)

        # Show bounding box and prediction on image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, char, (x, y+15), 0, 0.8, (0, 0, 255), 2)


    if num_chars > 0:
        average_prob = total_prob / num_chars
        print("Average confidence score of Resnet-50:", average_prob)
    else:
        print("No characters found for recognition.")

    vehicle_plate_easyocr = reader.readtext(image_org, detail=0)

    # Combine EasyOCR results into a single string
    vehicle_plate_easyocr = ''.join(vehicle_plate_easyocr)

    vehicle_plate_easyocr_n = correct_license_plate(vehicle_plate_easyocr)

    # Pytessract
    text_tesseract = pytesseract.image_to_string(image_ocr)
    tesseract_data = pytesseract.image_to_data(image_ocr, output_type=pytesseract.Output.DICT)
    tesseract_text = ''.join(tesseract_data['text'])
    tesseract_confidences = tesseract_data['conf']
    tesseract_avg_confidence = sum(tesseract_confidences) / len(tesseract_confidences)

    return vehicle_plate, vehicle_plate_easyocr_n.upper(), text_tesseract, tesseract_avg_confidence

#----------------------------------------------------------------

def get_state_and_state_name(license_plate_easyocr):
    state_dict = {
        "AN":"Andaman and Nicobar Islands",
        "AP": "Andhra Pradesh",
        "AR": "Arunachal Pradesh",
        "AS": "Assam",
        "BR": "Bihar",
        "CG": "Chhattisgarh",
        "DN": "Dadra and Nagar Haveli",
        "DD": "Daman and Diu",
        "DL": "Delhi",
        "GA": "Goa",
        "GJ": "Gujarat",
        "HR": "Haryana",
        "HP": "Himachal Pradesh",
        "JK": "Jammu and Kashmir",
        "JH": "Jharkhand",
        "KA": "Karnataka",
        "KL": "Kerala",
        "LA": "Ladakh",
        "LD": "Lakshadweep",
        "MP": "Madhya Pradesh",
        "MH": "Maharashtra",
        "MN": "Manipur",
        "ML": "Meghalaya",
        "MZ": "Mizoram",
        "NL": "Nagaland",
        "OD": "Odisha",
        "PB": "Punjab",
        "PY":"Pondicherry",
        "RJ": "Rajasthan",
        "SK": "Sikkim",
        "TN": "Tamil Nadu",
        "TS": "Telangana",
        "TR": "Tripura",
        "UP": "Uttar Pradesh",
        "UK": "Uttarakhand",
        "WB": "West Bengal"
    }

    state_code = license_plate_easyocr[:2].upper()
    state = state_dict.get(state_code, "Unknown State")
    return state_code, state

def check_and_get_tag(mycursor, license_plate):
    sql = "SELECT tag_name FROM tags WHERE license_plate = %s"
    val = (license_plate,)
    mycursor.execute(sql, val)
    tag_entry = mycursor.fetchone()

    if tag_entry:
        return tag_entry[0]  # Return the tag name
    else:
        return "Visitor"  # Default tag for unknown plates


def store_readings_to_mysql(results, mycursor, mydb):
    # Get current date and time
    today = date.today()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    # Get weekday and month names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    day = weekday_names[now.weekday()]
    month = month_names[now.month - 1]

    # Prepare SQL statements for single insertion
    sql_readings = """
    INSERT INTO readings 
    (car_id, license_plate_resnet, license_plate_easyocr, license_plate_tes, confidence_score)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    license_plate_resnet = VALUES(license_plate_resnet),
    license_plate_easyocr = VALUES(license_plate_easyocr),
    license_plate_tes = VALUES(license_plate_tes),
    confidence_score = CASE
        WHEN VALUES(confidence_score) > confidence_score THEN VALUES(confidence_score)
        ELSE confidence_score
    END
    """

    sql_anpr = """
    INSERT INTO anpr 
    (Licence, StateCode, State, Date, Day, Month, Time, Tag)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    for frame_nmr, readings in results.items():
        for car_id, data in readings.items():
            confidence_score = data['license_plate']['text_score']
            license_plate_resnet = data['license_plate']['text_resnet']
            license_plate_easyocr = data['license_plate']['text_easyocr']
            license_plate_tes = data['license_plate']['text_tes']

            # Check if there's an existing entry for the car_id
            mycursor.execute("SELECT confidence_score FROM readings WHERE car_id = %s", (car_id,))
            existing_score = mycursor.fetchone()

            if existing_score is None or confidence_score > existing_score[0]:
                # Get state and state code
                state_code, state = get_state_and_state_name(license_plate_tes)

                # Determine the tag (Staff or Unknown)
                tag = check_and_get_tag(mycursor, license_plate_tes)

                # Insert or update data into readings table
                mycursor.execute(sql_readings, (car_id, license_plate_resnet, license_plate_easyocr, license_plate_tes, confidence_score))
                mydb.commit()

                # Insert data into anpr table
                mycursor.execute(sql_anpr, (license_plate_tes, state_code, state, today, day, month, current_time, tag))
                mydb.commit()

    return True, "Readings stored successfully in MySQL database."





