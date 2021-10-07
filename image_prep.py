import cv2
import numpy as np
import os

def conditional_save(image, save_to: str = None):
    '''Save an image to disk.

    Args:
        image (cv2 image): image to save to disk
        save_to (str): path to save the image to. does not save if equals None. default=None
    '''
    if save_to:
        cv2.imwrite(save_to, image)

def grayscale(image, save_to: str = None):
    '''Make the image grayscale.

    Args:
        image (cv2 image): base image to convert
        save_to (str): path to save the image, does not save if it equals None. default=None
    
    Returns:
        processed image in cv2 image format
    '''
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conditional_save((img, save_to))
    return img

def black_and_white(image, threshold: int = 100, maxval: int = 255, save_to: str = None):
    '''Make the image black and white.

    Args:
        image (cv2 image): base image to convert
        threshold (int): threshold to pass to OpenCV, default=100
        maxval (int): maxval to pass to OpenCV, default=255
        save_to (str): path to save the image, does not save if it equals None. default=None

    Returns:
        processed image in cv2 image format
    '''
    _, img = cv2.threshold(image, threshold, maxval, cv2.THRESH_BINARY)
    conditional_save((img, save_to))
    return img

def remove_noise(image, kernel_size: 'tuple[int, int]' = (1, 1), dilate_iterations: int = 1, erode_iterations: int = 1, median_blur_k: int = 3, save_to: str = None):
    '''Remove noisy pixels from an image.

    Args:
        image (cv2 image): base image to process
        kernel_size (int): kernel_size to pass to OpenCV, default=(1,1)
        dilate_iterations (int): iterations for the dilation process to pass to OpenCV, default=1
        erode_iterations (int): iterations for the erosion process to pass to OpenCV, default=1
        median_blur_k (int): k-size for the medianBlur to passo to OpenCV, default=3
        save_to (str): path to save the image, does not save if it equals None. default=None

    Returns:
        processed image in cv2 image format
    '''
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.dilate(image, kernel, iterations=dilate_iterations)
    image = cv2.erode(image, kernel, iterations=erode_iterations)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, median_blur_k)
    conditional_save((image, save_to))
    return image

def prepare_image(image, output_path: str = None, temp_folder: str = None, black_and_white: bool= True, remove_noise: bool= False, verbose: bool= False):
    '''
    Apply selected preparations to an image.

    Args:
        image (cv2 image): base image to process
        output_path (str): path to save the final image, does not save if equals None default=None
        temp_folder (str): path to save intermediary images, does not save if equals None default=None
        black_and_white (bool): flag to convert the image to black and white, default=True
        remove_noise (bool): flag to remove noise from the image, default=False
        verbose (bool): print extra information to console?
    
    Returns:
        processed image in cv2 image format
    '''
    if black_and_white:
        save_to = os.path.join(temp_folder, 'grayscale.png') if temp_folder else None
        if verbose:
            print('converting to grayscale...', f'saving temp file to "{save_to}"' if save_to else '')
        image = grayscale(image, save_to)

        save_to = os.path.join(temp_folder, 'black_and_white.png') if temp_folder else None
        if verbose:
            print('converting to black and white...', f'saving temp file to "{save_to}"' if save_to else '')
        image = black_and_white(image, save_to)
    
    if remove_noise:
        save_to = os.path.join(temp_folder, 'remove_noise.png') if temp_folder else None
        if verbose:
            print('removing pixel noise...', f'saving temp file to "{save_to}"' if save_to else '')
        image = remove_noise(image, save_to)
    
    if output_path:
        if verbose:
            print('saving final image to "{output_path}"')
        cv2.imwrite(image, output_path)
    
    return image
