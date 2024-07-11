import cv2
import numpy as np
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils

# Constants
RATIO_PIXEL_TO_CM = 78  # Default: 78 pixels are 1 cm (adjust based on image scaling)
RATIO_PIXEL_TO_SQUARE_CM = RATIO_PIXEL_TO_CM ** 2

def preprocess_image(image_path):
    """ Preprocesses the image by reading and converting to RGB. """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Image file not found or unable to read.")
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def detect_and_segment_objects(image, predictor):
    """ Detects and segments objects using Detectron2's Mask R-CNN model. """
    # Run prediction
    outputs = predictor(image)
    instances = outputs["instances"]
    return instances

def filter_out_background(instances, min_confidence=0.5):
    """ Filters out background by removing instances with low confidence scores. """
    # Filter instances based on confidence score
    high_confidence_idx = instances.scores > min_confidence
    return instances[high_confidence_idx]

def apply_colored_mask(image, mask, color):
    """ Applies a colored mask to the image. """
    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    # Blend mask with original image
    return cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

def calculate_and_annotate_areas(image, instances, pixel_per_cm):
    """ Calculates and annotates areas of segmented objects. """
    annotated_image = image.copy()
    masks = instances.pred_masks.cpu().numpy()  # Get masks
    boxes = instances.pred_boxes.tensor.cpu().numpy()  # Get bounding boxes
    scores = instances.scores.cpu().numpy()  # Get confidence scores

    # Define colors for annotations
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    green_color = (0, 255, 0)

    for idx in range(masks.shape[0]):
        mask = masks[idx]
        box = boxes[idx]
        score = scores[idx]

        # Calculate area in pixels and convert to cm^2
        area_pixels = np.sum(mask)
        area_cm2 = area_pixels / RATIO_PIXEL_TO_SQUARE_CM

        # Classify objects based on area thresholds
        if area_cm2 < 100:
            category = "Small Object"
        elif area_cm2 < 500:
            category = "Medium Object"
        else:
            category = "Large Object"

        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_color = green_color if category == "Large Object" else colors[idx % len(colors)]

        # Apply colored mask to the interior
        annotated_image = apply_colored_mask(annotated_image, mask, contour_color)

        # Draw contours on the image
        cv2.drawContours(annotated_image, contours, -1, contour_color, 2)

        # Annotate area and category
        x1, y1, x2, y2 = map(int, box)
        cv2.putText(annotated_image, f"A: {area_cm2:.2f} cm^2", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, contour_color, 2)
        cv2.putText(annotated_image, category, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, contour_color, 2)

    return annotated_image

def find_reference_object(contours):
    """ Finds the reference object from the contours. """
    ref_object = contours[0]
    # Find minimum area bounding box
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    # Calculate distance in pixels between two points
    dist_in_pixel = euclidean(tl, tr)
    # Known reference object size in cm
    known_dist_in_cm = 2  
    # Calculate pixels per cm
    pixel_per_cm = dist_in_pixel / known_dist_in_cm
    return pixel_per_cm

def main(image_path):
    """ Main function to perform instance segmentation, calculate areas, and annotate results. """
    # Load the Detectron2 model configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # Set lower confidence threshold to capture more objects
    cfg.MODEL.DEVICE = "cuda"  # Run model on GPU
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    
    # Preprocess the input image
    image = preprocess_image(image_path)

    # Convert image to grayscale and apply Gaussian blur for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detect edges using Canny edge detection
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours in the edged image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours that are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    # Find reference object for scale based on contours
    pixel_per_cm = find_reference_object(cnts)

    # Detect and segment objects using Mask R-CNN
    instances = detect_and_segment_objects(image, predictor)
    
    # Filter out background objects based on confidence threshold
    instances = filter_out_background(instances, min_confidence=0.3)  # Adjust confidence threshold as needed
    
    # Calculate object areas and annotate the image
    annotated_image = calculate_and_annotate_areas(image, instances, pixel_per_cm)
    
    # Display the annotated image
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = "Image-Path/image.jpg/png/jpeg"  # Replace with your image path
    main(image_path)
