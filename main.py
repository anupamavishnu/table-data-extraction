import TableExtractor as te
import TableLinesRemover as tlr
import pytesseract
import cv2
path_to_image = r"C:\Drive-D\PROJECT_SAMPLE\LUMINAR_PROJECT\nutrition_table.jpg"
table_extractor = te.TableExtractor(path_to_image)
input_image = table_extractor.execute()
#cv2.imshow("Input_image",input_image)
lines_remover = tlr.TableLinesRemover(input_image)
image_without_lines = lines_remover.execute()
cv2.imshow("image_without_lines", image_without_lines)
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
text=pytesseract.image_to_string(image_without_lines)
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()