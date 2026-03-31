import cv2
import numpy as np
import os

TEMPLATE_DIR = "template"
TEST_DIR = "test"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# 1. ALIGN IMAGES (important)
# ----------------------------------
def align_images(template, test):
    gray1 = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:200]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(test, H, (template.shape[1], template.shape[0]))

    return aligned

# ----------------------------------
# 2. FIND PCB ROI (Contour Based)
# ----------------------------------
def extract_pcb_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)

    roi = cv2.bitwise_and(image, image, mask=mask)
    return roi, mask

# ----------------------------------
# 3. ROI IMAGE SUBTRACTION
# ----------------------------------
def roi_subtract(template_roi, test_roi):
    diff = cv2.absdiff(template_roi, test_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return gray

# ----------------------------------
# 4. THRESHOLD + CLEAN
# ----------------------------------
def get_defect_mask(diff):
    blur = cv2.GaussianBlur(diff, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.dilate(clean, kernel, iterations=1)

    return clean

# ----------------------------------
# 5. HIGHLIGHT DEFECTS
# ----------------------------------
def highlight(original, mask):
    result = original.copy()
    result[mask == 255] = [0,0,255]
    return result

# ----------------------------------
# MAIN LOOP
# ----------------------------------
for file in os.listdir(TEMPLATE_DIR):

    template_path = os.path.join(TEMPLATE_DIR, file)
    test_path = os.path.join(TEST_DIR, file)

    if not os.path.exists(test_path):
        continue

    template = cv2.imread(template_path)
    test = cv2.imread(test_path)

    test = cv2.resize(test, (template.shape[1], template.shape[0]))

    aligned = align_images(template, test)

    # Extract ROI from template (reference)
    template_roi, roi_mask = extract_pcb_roi(template)
    test_roi = cv2.bitwise_and(aligned, aligned, mask=roi_mask)

    diff = roi_subtract(template_roi, test_roi)
    defect_mask = get_defect_mask(diff)
    result = highlight(aligned, defect_mask)

    # Save outputs
    cv2.imwrite(f"{OUTPUT_DIR}/{file}_roi.png", template_roi)
    cv2.imwrite(f"{OUTPUT_DIR}/{file}_diff.png", diff)
    cv2.imwrite(f"{OUTPUT_DIR}/{file}_mask.png", defect_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{file}_result.png", result)

    print("Processed:", file)

print("ROI subtraction completed.")
