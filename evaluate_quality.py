import os
import re
import glob
import pandas as pd
from unidecode import unidecode
from nltk.metrics.distance import edit_distance

import utils

def count_characters(output_path='quality.tsv'):
    all_files = []

    editions = glob.glob('./output/*')
    for ed in editions:
        ed_name = utils.get_name(ed, 0).lower()
        pages = glob.glob(f'{ed}/*')
        for page in pages:
            page_name = utils.get_name(page, 0)
            all_files.append((ed_name, page_name, page))
        
    df = pd.DataFrame(columns=['edition', 'page', 'base', 'grayscale', 'processed'])

    for ed_name, page_name, page in all_files:
        base_path = os.path.join(page, 'base.txt')
        gray_path = os.path.join(page, 'gray.txt')
        proc_path = os.path.join(page, 'processed.txt')
        results = { 'edition': ed_name, 'page': page_name, 'base': 0, 'grayscale': 0, 'processed': 0 }
        
        with open(base_path, 'r', encoding='utf-8') as f:
            text = unidecode(f.read())
            text = re.sub(r'[^\w]+', '', text)
            results['base'] = len(text)
        
        with open(gray_path, 'r', encoding='utf-8') as f:
            text = unidecode(f.read())
            text = re.sub(r'[^\w]+', '', text)
            results['grayscale'] = len(text)

        with open(proc_path, 'r', encoding='utf-8') as f:
            text = unidecode(f.read())
            text = re.sub(r'[^\w]+', '', text)
            results['processed'] = len(text)
        
        df = df.append(results, ignore_index=True)

    df.to_csv(output_path, index=False, sep='\t')

def char_accuracy(ground_truth: str, recognized: str, ignore_accents: bool = True, ignore_newline: bool = True, ignore_symbols: bool = True) -> float:
    ground_truth = ground_truth.lower()
    recognized = recognized.lower()
    if ignore_accents:
        ground_truth = unidecode(ground_truth)
        recognized = unidecode(recognized)
    
    if ignore_newline:
        ground_truth = re.sub(r'\n+', ' ', ground_truth)
        recognized = re.sub(r'\n+', ' ', recognized)

    if ignore_symbols:
        ground_truth = re.sub(r'[^A-Za-z0-9,.]', '', ground_truth)
        recognized = re.sub(r'[^A-Za-z0-9,.]', '', recognized)

    m = len(ground_truth)
    d = edit_distance(ground_truth, recognized)
    return max(0, (m - d) / m)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ground-truth', '-gt', required=True)
    parser.add_argument('--ocr', '-o', required=True)
    args = parser.parse_args()
    
    with open(args.ground_truth, 'r', encoding='utf8') as f:
        gt = f.read()
    
    with open(args.ocr, 'r', encoding='utf8') as f:
        ocr = f.read()
    
    print(char_accuracy(gt, ocr))
