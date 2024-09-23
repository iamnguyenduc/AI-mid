Trâm Anh làm đy

import cv2
import numpy as np

# Step 1: Load the training images (positive and negative samples)
positive_image_path = 'pos2.png' 
negative_image_path = 'neg.png'  
input_image_path = 'in3.png'        

# Load the images and resize them to 180x381
positive_image = cv2.imread(positive_image_path, cv2.IMREAD_GRAYSCALE)
negative_image = cv2.imread(negative_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Resize the input image to ensure it's large enough for sliding window detection
window_width, window_height = 180, 381

# If your input image is not already grayscale, convert it
if len(positive_image.shape) == 3:
    positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2GRAY)
if len(negative_image.shape) == 3:
    negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2GRAY)
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Step 2: Preprocessing - Convert to Binary (Thresholding)
_, positive_image_binary = cv2.threshold(positive_image, 128, 255, cv2.THRESH_BINARY)
_, negative_image_binary = cv2.threshold(negative_image, 128, 255, cv2.THRESH_BINARY)
_, input_image_binary = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)

# Step 3: Sliding Window for Object Detection
def sliding_window(input_image, window_width, window_height, step_size):
    # Slide a window across the image
    for y in range(0, input_image.shape[0] - window_height + 1, step_size):
        for x in range(0, input_image.shape[1] - window_width + 1, step_size):
            yield (x, y, input_image[y:y + window_height, x:x + window_width])

# Step 4: Mismatch Count and Detection
def count_mismatches(window, template_image):
    # Calculate the number of mismatches (pixel-wise comparison)
    return np.sum(window != template_image)

# Define step size for sliding window
step_size = 10  # Adjust step size for speed vs accuracy trade-off

# Initialize variables to track the best match
best_match_position = None
min_mismatch_count = float('inf')

for (x, y, window) in sliding_window(input_image_binary, window_width, window_height, step_size):
    if window.shape[0] == window_height and window.shape[1] == window_width:
        # Count mismatches with positive and negative images
        positive_mismatch_count = count_mismatches(window, positive_image_binary)
        negative_mismatch_count = count_mismatches(window, negative_image_binary)
        
        # If the mismatch with positive image is below the threshold, update the best match
        if positive_mismatch_count < min_mismatch_count:
            min_mismatch_count = positive_mismatch_count
            best_match_position = (x, y)

# Step 5: Display Result
if best_match_position is not None:
    # Draw a rectangle around the detected object
    output_image = cv2.imread(input_image_path)
    cv2.rectangle(output_image, best_match_position, 
                  (best_match_position[0] + window_width, best_match_position[1] + window_height), 
                  (0, 255, 0), 2)
    
    # Display the output image with detected object
    cv2.imshow("Detected Object", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No object detected.")

# Step 6: Save the result (optional)
# cv2.imwrite('detected_output.jpg', output_image)

