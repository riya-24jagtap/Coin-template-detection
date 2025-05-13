import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog

# print("Select the originl image:")
# original_path = filedialog.askopenfilename(title="Select Original Image")
# print("Select the template image:")
# template_path = filedialog.askopenfilename(title="Select Template Image")
# Load the main image and the template image

orginal_img = cv2.imread(r"C:/Users/RUCHI/Documents/riya/coin_img.jpg")
template_img = cv2.imread(r"C:/Users/RUCHI/Documents/riya/coin_temp2.jpg")

if orginal_img is None or template_img is None:
    print("Error: Could not load the image or template. Please check the file paths.")
    exit()
gray_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
# Show the template image in color
plt.figure(figsize=(6, 5))
plt.title("Template Image")
plt.imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))  
plt.axis('off') 
plt.show()  
#temp dimensio ns
w, h = gray_template.shape[::-1]
print("Template size:", w, "x", h)
result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED) #normalized cross-correlation coefficient

#Get the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
threshold = 0.80
if max_val >= threshold:
    #draws green rectangle around the match
    top_left = max_loc 
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(orginal_img, top_left, bottom_right, (0, 0, 0), 9)
plt.figure(figsize=(12, 6))
plt.title("Matched Result")
plt.imshow(cv2.cvtColor(orginal_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
