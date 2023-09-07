import cv2
import numpy as np
class TableLinesRemover:
    def __init__(self, image):
        self.image = image

    def execute(self):
        self.grayscale_image()
        self.threshold_image()
        self.invert_image()
        self.erode_vertical_lines()
        self.erode_horizontal_lines()
        self.combine_eroded_images()
        self.dilate_combined_image_to_make_lines_thicker()
        self.subtract_combined_and_dilated_image_from_original_image()
        self.remove_noise_with_erode_and_dilate()
        return self.image_without_lines_noise_removed
    
    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray_scale_image",self.grey)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grey, 127, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("Thresholded_image",self.thresholded_image)

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)
        #cv2.imshow("Inverted_image",self.inverted_image)

    def erode_vertical_lines(self):
        hor = np.array([[1,1,1,1,1,1]])
        self.vertical_lines_eroded_image = cv2.erode(self.inverted_image, hor, iterations=10)
        self.vertical_lines_eroded_image = cv2.dilate(self.vertical_lines_eroded_image, hor, iterations=10)
        #cv2.imshow("Vertical_lines_eroded_image",self.vertical_lines_eroded_image)

    def erode_horizontal_lines(self):
        ver = np.array([[1],
               [1],
               [1],
               [1],
               [1],
               [1],
               [1]])
        self.horizontal_lines_eroded_image = cv2.erode(self.inverted_image, ver, iterations=10)
        self.horizontal_lines_eroded_image = cv2.dilate(self.horizontal_lines_eroded_image, ver, iterations=10)
        #cv2.imshow("Horizontal_lines_eroded_image",self.horizontal_lines_eroded_image)

    def combine_eroded_images(self):
        self.combined_image = cv2.add(self.vertical_lines_eroded_image, self.horizontal_lines_eroded_image)
       # cv2.imshow("Combined_image",self.combined_image)

    def dilate_combined_image_to_make_lines_thicker(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.combined_image_dilated = cv2.dilate(self.combined_image, kernel, iterations=5)
        #cv2.imshow("combined_image_dilated",self.combined_image_dilated)

    def subtract_combined_and_dilated_image_from_original_image(self):
        self.image_without_lines = cv2.subtract(self.inverted_image, self.combined_image_dilated)
        #cv2.imshow("Subtrcted_img",self.image_without_lines)

    def remove_noise_with_erode_and_dilate(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image_without_lines_noise_removed = cv2.erode(self.image_without_lines, kernel, iterations=1)
        self.image_without_lines_noise_removed = cv2.dilate(self.image_without_lines_noise_removed, kernel, iterations=1)

