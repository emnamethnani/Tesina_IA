import cv2

# Load image
img = cv2.imread('C:/Users/Admin/Desktop/projet_python/premier programme/image1.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
canny = cv2.Canny(blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and find license plate candidate
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    x, y, w, h = cv2.boundingRect(approx)

    if len(approx) == 4 and w > 100 and h > 30:
        plate = img[y:y + h, x:x + w]
        cv2.imshow('License Plate', plate)
        cv2.waitKey(0)
        break