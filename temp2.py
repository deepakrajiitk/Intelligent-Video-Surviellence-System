import cv2
import numpy as np
from collections import Counter

# Load the image
car_image = cv2.imread('red.jpg')

# Apply GrabCut to separate the car from the background
mask = np.zeros(car_image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (1, 1, car_image.shape[1]-1, car_image.shape[0]-1)
cv2.grabCut(car_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create a binary mask of the segmented car
mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')


# Apply the binary mask to the car image
segmented_car = car_image * mask[:,:,np.newaxis]
cv2.imwrite("checkcar.jpg", segmented_car)

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(segmented_car, cv2.COLOR_BGR2HSV)

# Define the color ranges for each color
color_ranges = [
    ((0, 70, 50), (10, 255, 255), "red"),   # red
    ((20, 70, 50), (30, 255, 255), "orange"), # orange
    ((36, 70, 50), (70, 255, 255), "yellow"), # yellow
    ((80, 70, 50), (100, 255, 255), "green"), # green
    ((110, 70, 50), (130, 255, 255), "blue"), # blue
    ((140, 70, 50), (160, 255, 255), "purple"), # purple
    ((0, 0, 200), (180, 30, 255), "white"),  # white
    ((0, 0, 0), (180, 255, 30), "black"),    # black
    ((0, 0, 0), (180, 25, 150), "gray"),    # gray
    ((10, 30, 30), (20, 255, 255), "brown")  # brown
]

# Threshold the image for each color and count the number of pixels
color_pixels = []
for (lower, upper, color) in color_ranges:
    mask = cv2.inRange(hsv_image, lower, upper)
    color_pixels.append(cv2.countNonZero(mask))

# Get the ten most common colors
color_counts = Counter(dict(zip([color for (lower, upper, color) in color_ranges], color_pixels)))
top_colors = color_counts.most_common(10)

# Print the ten most common colors and their pixel counts
for color, count in top_colors:
    print(f"{color}: {count} pixels")
