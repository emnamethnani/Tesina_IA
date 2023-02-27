import cv2

# Load the image
img = cv2.imread('C:/Users/Admin/Desktop/projet_python/premier programme/image1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding to binarize the image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours and get the largest contour
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
plate_contour = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        plate_contour = approx
        break

# Draw the license plate contour on the original image
if plate_contour is not None:
    cv2.drawContours(img, [plate_contour], 0, (0, 255, 0), 2)

# Display the image with the license plate contour
cv2.imshow('License Plate Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


