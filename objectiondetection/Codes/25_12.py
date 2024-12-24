import cv2
import numpy as np
import os
import math
from ultralytics import YOLO

# Paths
image_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\images'
image_labels_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\labels'
calibration_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\calib'
output_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\results'
os.makedirs(output_folder_path, exist_ok=True)

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1: List or tuple with coordinates [x_min, y_min, x_max, y_max].
        box2: List or tuple with coordinates [x_min, y_min, x_max, y_max].
    Returns:
        IoU value as a float.
    """
    # Coordinates of the intersection rectangle
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Intersection area
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Areas of the individual bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # IoU
    iou = intersection_area / union_area
    return iou

def load_intrinsic_matrix(calibration_file_path):
    try:
        intrinsic_matrix = np.loadtxt(calibration_file_path)
        return intrinsic_matrix
    except Exception as e:
        print(f"Error loading intrinsic matrix from {calibration_file_path}: {e}")
        return None

# Calculate distance based on bounding box and intrinsic parameters
def calculate_distance_aligned(intrinsic, bbox, camera_height=1.65):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_min, y_min, x_max, y_max = bbox
    c1 = (x_min, y_min)
    c2 = (x_max, y_min)
    c3 = (x_max, y_max)
    c4 = (x_min, y_max)

    # Midpoints
    m1 = ((x_min + x_max) / 2, y_min)
    m2 = (x_max, (y_min + y_max) / 2)
    m3 = ((x_min + x_max) / 2, y_max)
    m4 = (x_min, (y_min + y_max) / 2)

    # Combine all points
    points = [c1, c2, c3, c4, m1, m2, m3, m4]
    distances = []  # Store distances for all points
    for (u, v) in points:
        try:
            Y = (camera_height * fy) / (v - cy)  # Depth along Y-axis
            X = (u - cx) * Y / fx  # X-coordinate in world space
            distance = np.sqrt(X ** 2 + camera_height ** 2 + Y ** 2)
            distances.append(distance)
        except ZeroDivisionError:
            print(f"Division by zero for point ({u}, {v}). Check camera parameters or bounding box.")
            distances.append(float('inf'))  # Add infinity if division by zero occurs

    # Return the minimum distance among all points
    return min(distances)

model = YOLO("yolo11x.pt")

# Loop through images
for filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, filename)
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {filename}")
        continue

    results = model.predict(source=image_path, conf=0.5)
    ground_truth_boxes = []
    detected_boxes = []
    car_id_counter = 0  # Counter for unique car IDs

    # Process YOLO detections
    for result in results:
        for box in result.boxes:
            if box.cls[0] == 2:  # Class ID for "car"
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO format conversion
                detected_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Process ground truth labels
    label_file_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
    label_file_path = os.path.join(image_labels_path, label_file_name)
    if os.path.exists(label_file_path):
        with open(label_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) < 6:
                    continue
                class_id, x_min, y_min, x_max, y_max, GT_distance = data[:6]
                x_min, y_min, x_max, y_max = map(int, map(float, [x_min, y_min, x_max, y_max]))
                GT_distance = float(GT_distance)
                ground_truth_boxes.append([x_min, y_min, x_max, y_max, GT_distance])
                cv2.rectangle(img, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 2)  # Draw GT box

    output_text_path = os.path.join(output_folder_path, f"results_{filename}.txt")
    with open(output_text_path, "w") as output_file:
        # Match detected cars with ground truth boxes
        for detected_box in detected_boxes:
            for gt_index, ground_truth_box in enumerate(ground_truth_boxes):
                iou = calculate_iou(detected_box, ground_truth_box[:4])
                if iou > 0.5:  # Threshold for matching
                    calibration_file_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
                    calibration_file_path = os.path.join(calibration_folder_path, calibration_file_name)

                    # Load intrinsic matrix and calculate distance
                    if os.path.exists(calibration_file_path):
                        intrinsic_matrix = load_intrinsic_matrix(calibration_file_path)
                        if intrinsic_matrix is not None:
                            distance_car = calculate_distance_aligned(intrinsic_matrix, detected_box)
                        else:
                            distance_car = float('inf')
                    else:
                        print(f"Calibration file not found: {calibration_file_path}")
                        distance_car = float('inf')

                    x1, y1, x2, y2 = detected_box
                    car_id_counter += 1
                    output_file.write(
                        f"CAR ID: {car_id_counter}, YOLO distance: {distance_car:.2f}m, GT distance: {ground_truth_box[4]:.2f}m, IoU: {iou:.2f}\n")
                    print(
                        f' CAR ID:{car_id_counter} Yolo distance: {distance_car:.2f}m, GT distance: {ground_truth_box[4]:.2f}m, IoU: {iou:.2f}')

                    # Annotate image with bounding box, ID, YOLO and GT distances

                    # Text positions and sizes
                    text_scale = 0.3
                    text_thickness = 1
                    id_text = f"ID: {car_id_counter}"
                    yolo_text = f"YOLO: {distance_car:.2f}m"
                    gt_text = f"GT: {ground_truth_box[4]:.2f}m"

                    # ID background
                    id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
                    cv2.rectangle(img, (x1, y1 - 40), (x1 + id_size[0] + 5, y1 - 20), (0, 0, 0), -1)
                    cv2.putText(img, id_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255),
                                text_thickness)

                    # YOLO background
                    yolo_size = cv2.getTextSize(yolo_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + yolo_size[0] + 5, y1 - 5), (0, 0, 0), -1)
                    cv2.putText(img, yolo_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255),
                                text_thickness)

                    # GT background
                    gt_size = cv2.getTextSize(gt_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
                    cv2.rectangle(img, (x1, y1), (x1 + gt_size[0] + 5, y1 + 15), (0, 0, 0), -1)
                    cv2.putText(img, gt_text, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255),
                                text_thickness)

                    break

        # Save annotated image
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, img)
        print(f"Results saved to {output_path}")

