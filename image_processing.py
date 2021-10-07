from numpy.lib.npyio import save
from utils import display
import cv2
import os

def detect_main_body(image, output_path: str = None, temp_folder: str = None):
    '''Detect the main body of text, discarding margins and other elements.

    Args:
        image (cv2 image): black and white image to process
        output_path (str): path to write the output image to, does not save if equals None. default=None
        temp_folder (str): folder to write the intermediary files to, does not save if equals None. default=None

    Returns:
        image of the main body (cv2 image)
    '''
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 75))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    if temp_folder:
        save_to = os.path.join(temp_folder, 'main_body/dilate.png')
        cv2.imwrite(save_to, dilate)
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    cnt = cnts[-1] # select largest
    x, y, w, h = cv2.boundingRect(cnt)

    img_main = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, img_main)

    return img_main

def detect_columns(image, output_folder: str = None, temp_folder: str = None):
    '''Detect columns of text in an image.

    Args:
        image (cv2 image): black and white image to process
        output_folder (str): path to folder in which to save the image of detected columns, does not save if equals None. default=None
        temp_folder (str): path to folder in which to intermediary images, does not save if equals None. default=None

    Returns:
        tuple[list, list]: a tuple containing two elements:
            1. A list of the detected column images
            2. A list of the rectangle coordinates for each column (x, y, w, h)
    '''
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    if temp_folder:
        save_to = os.path.join(temp_folder, 'columns/dilate.png')
        cv2.imwrite(save_to, dilate)
    
    def contains(columns: 'list[tuple[int, int, int, int]]', x: int, y: int, w: int, h: int):
        '''Checks if the rectangle is already fully contained in a column.
        
        Args:
            columns (list): coordinates of the columns already detected
            x (int): x coordinate in pixels to check
            y (int): y coordinate in pixels to check
            w (int): width in pixels of the rectangle
            h (int): height in pixels of the rectangle
        
        Returns:
            bool: True if the rectangle (x, y, w, h) is fully contained in a rectangle in the columns list
        '''
        for c in columns:
            x1, y1, w1, h1 = c
            if x >= x1 and x + w <= x1 + w1 and y >= y1 and y + h <= y1 + h1:
                return True
        
        return False

    column_images = []
    boxed_image = image.copy() if temp_folder else None
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    
    i = 0
    columns = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 200 and w > 150 and not contains(columns, x, y, w, h):
            columns.append((x, y, w, h))
            roi = image[y:y+h, x:x+w]
            
            if output_folder: 
                save_to = os.path.join(f'columns/roi_{i}.png', roi)
                cv2.imwrite(save_to, roi)
            
            column_images.append(roi)
            
            if temp_folder:
                cv2.rectangle(boxed_image, (x, y), (x+w, y+h), (36, 255, 12), 2)

            i += 1
    if temp_folder:
        save_to = os.path.join(temp_folder, 'column_boxes.png')
        cv2.imwrite(save_to, boxed_image)
    
    return column_images, columns
