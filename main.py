import glob
import cv2
import os

from process_pdfs import convert_pdfs
from image_prep import prepare_image
from image_processing import detect_main_body, detect_columns
import utils

PROCESS_PDFS = False
VERBOSE = True

def log(msg):
    if VERBOSE:
        print(msg)

if PROCESS_PDFS:
    log('converting PDFs into PNGs')
    convert_pdfs(glob.glob('./input/raw/*.pdf'), './input/processed', VERBOSE)

editions = glob.glob('./input/processed/*')
for ed in editions:
    ed_name = utils.get_name(ed, 0).lower()
    log(f'in edition "{ed_name}"')
    pages = glob.glob(f'{ed}/*.png')
    for page in pages:
        page_name = utils.get_name(page)
        
        log(f'...in page "{page_name}"')
        image = cv2.imread(page)
        
        os.makedirs(f'./temp/{ed_name}/{page_name}/columns', exist_ok=True)
        os.makedirs(f'./temp/{ed_name}/{page_name}/columns_temp', exist_ok=True)
        os.makedirs(f'./output/{ed_name}/{page_name}/columns', exist_ok=True)

        log('preparing image')
        image = prepare_image(image, f'./temp/{ed_name}_{page_name}.png', f'./temp/{ed_name}/{page_name}', verbose=VERBOSE)
        
        log('detecting the main body')
        image = detect_main_body(image, temp_folder=f'./temp/{ed_name}/{page_name}')
        
        log('detecting the columns')
        detect_columns(image, f'./temp/{ed_name}/{page_name}/columns', f'./temp/{ed_name}/{page_name}/columns_temp', verbose=VERBOSE)

        log('running OCR on the unprocessed page')
        utils.run_ocr(page, f'./output/{ed_name}/{page_name}/base.txt', verbose=VERBOSE)

        log('running OCR on the processed columns')
        utils.run_ocr_on_columns(glob.glob(f'./temp/{ed_name}/{page_name}/columns/*.png'), f'./output/{ed_name}/{page_name}/columns', f'./temp/{ed_name}/{page_name}/processed.txt')

        log(f'DONE with page "{page_name}"')
    log(f'DONE with edition "{ed_name}"')
