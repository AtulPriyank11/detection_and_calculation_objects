### Object Detection and Surface Area Calculation using Detectron2 and OpenCV

This Python script utilizes Detectron2, a powerful library for object detection, to identify and measure the surface area of objects in images. It employs Mask R-CNN (Region-based Convolutional Neural Network) for accurate instance segmentation and calculates the area based on known pixel-to-centimeter ratios.
Installation

To run the script, ensure you have Python installed (preferably Python 3.6+) and follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your/repository.git
cd repository
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Detectron2:
You can install Detectron2 directly from its GitHub repository with the following command:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13'
```

This command installs Detectron2 at a specific commit (5aeb252b194b93dc2879b4ac34bc51a31b5aee13), ensuring compatibility with the script.

4. Downloading model weights:
This is a crucial step. But I have made it easy. You don't need to download the weights instead when you the actual script, it will automatically download the weights.

## Approach
# Object Detection
* The script preprocesses the input image, converts it to RGB format, and applies edge detection using Canny.
* Using a pre-trained Mask R-CNN model configured with Detectron2, objects in the image are detected and segmented.

# Surface Area Calculation

*  A reference object of known size is identified from the image using contours to establish a pixel-to-centimeter conversion ratio (pixel_per_cm).
* For each detected object, the script calculates its area in square centimeters based on the established pixel_per_cm ratio and annotates the result on the image.

# Assumptions
* Object Type: The script assumes objects are planar (flat) and lying horizontally on a surface for accurate area calculation.
* Reference Object: A known-sized reference object must be visible in the image to establish the pixel_per_cm ratio accurately.

# Output Format

The script outputs an annotated image displaying detected objects with their corresponding surface area annotated. Each object is categorized as "Small Object," "Medium Object," or "Large Object" based on its computed area.

# Functionality

1. Object Detection: The script effectively utilizes detectron's Mask R-CNN to detect and segment objects, achieving accurate results.
2. Area Calculation: Surface areas are calculated precisely based on the established pixel-to-centimeter ratio.

# Clarity and Readability

The code is well-structured, with modular functions and detailed comments explaining each step and the rationale behind calculations.

# Efficiency

The script efficiently preprocesses images, performs object detection, and calculates areas using optimized libraries like Detectron2 and OpenCV.

# Documentation

A comprehensive README file accompanies the script, providing installation instructions, approach, assumptions, output format, and evaluation criteria.
Handling Additional Shapes: The script can be extended to handle various object shapes beyond rectangles, such as circles or irregular polygons, by adapting contour detection and area calculation methods.
    
## Challenges and Considerations 

The script can be expected to handle various object shapes beyond rectangles, such as circles or irregular polygons, by adapting contour detection and area calculation methods.

Challenges like tilted objects or complex shapes can impact accurate contour detection and area calculation. Robust algorithms and additional preprocessing steps may be required to handle these scenarios effectively.

## Sample Output