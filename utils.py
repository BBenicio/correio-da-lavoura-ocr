from matplotlib import pyplot as plt
from numpy.lib.function_base import disp

def display(im_path: str):
    '''Display the image using matplotlib.

    Use matplotlib to load and display an image. Good for viewing inline in notebooks.

    Args:
        im_path (str): path to the image to be displayed
    
    Remarks:
        https://stackoverflow.com/questions/28816046/
    '''
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

# Pre processing

import pdf2image
import glob
import os

def convert_pdfs(input_files: 'list[str]' = [], output_folder: str = './tmp', verbose: bool = False):
    '''Convert multiple PDF files into PNG images.

    Load PDFs and use pdf2image to convert each page into a different png image.

    Args:
        input_files (list): list of the paths to the PDF files to convert to PNG
        output_folder (str): destination folder to save the converted images
        verbose (bool): print extra information to console?
    
    Remarks:
        A directory is created inside the output folder for each input PDF,
        and the images correspondig to those PDF's pages are placed inside of
        said folder.

    '''
    def get_name(file_path):
        return file_path.replace('\\', '/').split('/')[-1][:-4]

    for file_path in input_files:
        name = get_name(file_path)
        if verbose: print('processing file:', name)
        
        output = os.path.join(output_folder, name)
        os.makedirs(output, exist_ok=True)
        
        pdf2image.convert_from_path(file_path, output_folder=output, output_file='page', poppler_path='C:/Misc/poppler-21.09.0/Library/bin', fmt='png')
        
        if verbose:
            out_files = [ get_name(f) for f in glob.glob(f'{output}/*.png') ]
            print(f'\tconverted {len(out_files)} pages:', ','.join(out_files))


# Processing

from PIL import Image
import pytesseract

def run_ocr(image_path: str, output_path: str = None, remove_spaces: bool = True, remove_hyphenation: bool = True, verbose: bool = False) -> str:
    '''Detect portuguese text from an image using pytesseract.

    Load an image from a path and run it through pytesseract to detect text.

    Args:
        image_path (str): path of input image
        output_path (str): path to write text output to, does not save if equals None. default=None
        remove_spaces (bool): flag to remove extra spaces in post-processing. default=True
        remove_hyphenation (bool): flag to remove hyphenation, joining words in post-processing. default=True
        verbose (bool): write extra information to console?

    Returns:
        str: text detected
    '''
    img = Image.open(image_path)
    if verbose:
        print(f'read image from "{image_path}"')
    
    result = pytesseract.image_to_string(img, lang='por')
    if verbose:
        print(f'detected {len(result)} characters in image')

    if remove_spaces:
        result = remove_extra_spaces(result)
        print(f'after space removal, got {len(result)} characters')
    
    if remove_hyphenation:
        result = treat_hyphenation(result)
        print(f'after hyphenation removal, got {len(result)} characters')

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            print(f'writing result to "{output_path}"')
            f.write(result)

def run_ocr_on_columns(columns_path: 'list[str]', temp_folder: str, output_path: str) -> str:
    '''Detect text from multiple images and append them together

    Args:
        columns_path (list[str]): list of paths to the images to process
        temp_folder (str): path to a directory to write the text files for each image
        output_path (str): path to a text file to write the final output

    Returns:
        str: All of the detected texts 
    '''
    for i in range(len(columns_path)):
        out = os.path.join(temp_folder, f'/columns/{i}.txt')
        run_ocr(columns_path[i], out)

    result = []
    for i in range(len(columns_path)):
        text_file = os.path.join(temp_folder, f'/columns/{i}.txt')
        with open(text_file, encoding='utf-8') as f:
            result.append(f.read())
    result = '\n\n'.join(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    
    return result

# Post processing

import re
def remove_extra_spaces(text: str) -> str:
    '''Remove double spaces and extra line feeds.

    Args:
        text (str): text to remove spaces from
    
    Returns:
        str: text with no double spaces or 3+ line feeds
    '''
    text = re.sub('\n{2,}', '\n\n', text)
    text = re.sub(' +', ' ', text)
    return text

def treat_hyphenation(text: str) -> str:
    '''Remove hyphenation from text.

    Args:
        text (str): text to remove hyphenation from
    
    Returns:
        str: text without hyphenation
    '''
    text = re.sub('-\n', '', text)
    return text