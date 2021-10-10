import glob
import cv2
import os
from unidecode import unidecode

from process_pdfs import convert_pdfs
from image_prep import prepare_image
from image_processing import crop_margins, detect_columns
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
    ed_name = unidecode(utils.get_name(ed, 0).lower())
    log(f'in edition "{ed_name}"')
    pages = glob.glob(f'{ed}/*.png')
    for page in pages:
        page_name = utils.get_name(page)
        
        log(f'...in page "{page_name}"')
        image = utils.load_image(page)
        
        os.makedirs(f'./temp/{ed_name}/{page_name}/columns', exist_ok=True)
        os.makedirs(f'./temp/{ed_name}/{page_name}/columns_temp', exist_ok=True)
        os.makedirs(f'./output/{ed_name}/{page_name}/columns', exist_ok=True)

        log('preparing image')
        pre_image = prepare_image(image, f'./temp/{ed_name}/pre_{page_name}.png', f'./temp/{ed_name}/{page_name}/pre', verbose=VERBOSE)
        
        log('detecting the main body')
        _, (x1, x2, y1, y2) = crop_margins(pre_image, f'./temp/{ed_name}/{page_name}/pre', f'./temp/{ed_name}/{page_name}/main.png')

        image = image[y1:y2, x1:x2]
        log('preparing image')
        image = prepare_image(image, f'./temp/{ed_name}/{page_name}.png', f'./temp/{ed_name}/{page_name}', verbose=VERBOSE)

        log('detecting the columns')
        detect_columns(image, f'./temp/{ed_name}/{page_name}/columns', f'./temp/{ed_name}/{page_name}/columns_temp', verbose=VERBOSE)

        log('running OCR on the unprocessed page')
        utils.run_ocr(page, f'./output/{ed_name}/{page_name}/base.txt', verbose=VERBOSE)

        log('running OCR on the grayscale page')
        utils.run_ocr(f'./temp/{ed_name}/{page_name}/grayscale.png', f'./output/{ed_name}/{page_name}/gray.txt', verbose=VERBOSE)

        log('running OCR on the processed columns')
        utils.run_ocr_on_columns(glob.glob(f'./temp/{ed_name}/{page_name}/columns/*.png'), f'./temp/{ed_name}/{page_name}/columns', f'./output/{ed_name}/{page_name}/processed.txt')

        log(f'DONE with page "{page_name}"')
    log(f'DONE with edition "{ed_name}"')
