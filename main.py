import glob
import os
import cv2
from unidecode import unidecode
from tqdm import tqdm
import pandas as pd

from process_pdfs import convert_pdfs
from image_prep import deskew, prepare_image
from image_processing import crop_background, crop_to_page, detect_columns
from evaluate_quality import count_characters
from mhs_layout_analisys import segment
import utils

PROCESS_PDFS = False
DO_OCR = True
OCR_BASE = DO_OCR and False
OCR_GRAY = DO_OCR and False
OCR_PROCESSED = DO_OCR and True
DO_MHS = False
VERBOSE = False

def log(msg):
    if VERBOSE:
        print(msg)

if PROCESS_PDFS:
    log('converting PDFs into PNGs')
    convert_pdfs(glob.glob('./input/raw/*.pdf'), './input/processed', VERBOSE)

all_files = []

editions = glob.glob('./input/processed/*')
for ed in editions:
    ed_name = unidecode(utils.get_name(ed, 0).lower())
    pages = glob.glob(f'{ed}/*.png')
    for page in pages:
        page_name = utils.get_name(page)
        all_files.append((ed_name, page_name, page))

for ed_name, page_name, page in tqdm(all_files):
    log(f'...in page "{page_name}" from "{ed_name}"')
    image = utils.load_image(page)
    
    os.makedirs(f'./temp/{ed_name}/{page_name}/columns', exist_ok=True)
    os.makedirs(f'./temp/{ed_name}/{page_name}/columns_temp', exist_ok=True)
    os.makedirs(f'./output/{ed_name}/{page_name}/columns', exist_ok=True)

    log('cropping image')
    image, _ = crop_background(image, f'./temp/{ed_name}/{page_name}/', f'./temp/{ed_name}/{page_name}/cropped.png')

    log('preparing image')
    image = prepare_image(image, f'./temp/{ed_name}/{page_name}/prepared.png', f'./temp/{ed_name}/{page_name}', verbose=VERBOSE)

    log('cropping page')
    image, _ = crop_to_page(image, f'./temp/{ed_name}/{page_name}/', f'./temp/{ed_name}/{page_name}/cropped_page.png')

    if DO_MHS:
        image, _, _ = segment(image, f'./temp/{ed_name}/{page_name}/')
        image = deskew(image)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        utils.conditional_save(image, f'./temp/{ed_name}/{page_name}/rotated_after_mhs.png')
        utils.conditional_save(image, f'./temp/{ed_name}/{page_name}.png')
    else:
        image = deskew(image)
        utils.conditional_save(image, f'./temp/{ed_name}/{page_name}.png')

    if OCR_BASE:
        log('running OCR on the unprocessed page')
        utils.run_ocr(page, f'./output/{ed_name}/{page_name}/base.xml', f'./temp/{ed_name}/{page_name}/tess_unproc.png', verbose=VERBOSE)

    if OCR_GRAY:
        log('running OCR on the grayscale page')
        utils.run_ocr(f'./temp/{ed_name}/{page_name}/grayscale.png', f'./output/{ed_name}/{page_name}/gray.xml', f'./temp/{ed_name}/{page_name}/tess_gray.png', verbose=VERBOSE)

    if OCR_PROCESSED:
        log('running OCR on the processed page')
        utils.run_ocr(f'./temp/{ed_name}/{page_name}.png', f'./output/{ed_name}/{page_name}/proc.xml', f'./temp/{ed_name}/{page_name}/tess_proc.png', verbose=VERBOSE)
    

    log(f'DONE with page "{page_name}" from "{ed_name}"')
