import glob
import os
from unidecode import unidecode
from tqdm import tqdm
import pandas as pd

from process_pdfs import convert_pdfs
from image_prep import prepare_image
from image_processing import crop_background, detect_columns
from evaluate_quality import count_characters
import utils

PROCESS_PDFS = False
VERBOSE = False

def log(msg):
    if VERBOSE:
        print(msg)

if PROCESS_PDFS:
    log('converting PDFs into PNGs')
    convert_pdfs(glob.glob('./input/raw/*.pdf'), './input/processed', VERBOSE)

confidence_scores = []
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
    image = prepare_image(image, f'./temp/{ed_name}/{page_name}.png', f'./temp/{ed_name}/{page_name}', verbose=VERBOSE)

    # log('detecting the columns')
    # detect_columns(image, f'./temp/{ed_name}/{page_name}/columns', f'./temp/{ed_name}/{page_name}/columns_temp', verbose=VERBOSE)

    log('running OCR on the unprocessed page')
    _, base = utils.run_ocr(page, f'./output/{ed_name}/{page_name}/base.txt', verbose=VERBOSE)

    log('running OCR on the grayscale page')
    _, gray = utils.run_ocr(f'./temp/{ed_name}/{page_name}/grayscale.png', f'./output/{ed_name}/{page_name}/gray.txt', verbose=VERBOSE)

    log('running OCR on the processed page')
    _, proc = utils.run_ocr(f'./temp/{ed_name}/{page_name}.png', f'./output/{ed_name}/{page_name}/proc.txt', verbose=VERBOSE)

    # log('running OCR on the columns')
    # _, processed = utils.run_ocr_on_columns(glob.glob(f'./temp/{ed_name}/{page_name}/columns/*.png'), f'./temp/{ed_name}/{page_name}/columns', f'./output/{ed_name}/{page_name}/processed.txt')

    confidence_scores.append({ 'edition': ed_name, 'page': page_name, 'base': base, 'grayscale': gray, 'processed': proc })

    log(f'DONE with page "{page_name}" from "{ed_name}"')

df = pd.DataFrame(confidence_scores)
df.to_csv('quality.tsv', index=False, sep='\t')