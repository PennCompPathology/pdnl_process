#!/usr/bin/env python3

import os
import sys
import argparse
import json

import numpy as np
from tqdm import tqdm
import pdnl_sana.geo
import pdnl_sana.image
import pdnl_sana.process
import pdnl_sana.logging
import pdnl_sana.filter
import pdnl_sana.threshold

from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to image data to process", required=True)
    parser.add_argument('-o', '--output_path', help="path to write outputs to", required=True)
    parser.add_argument('-s', '--staining_code', help="stain protocol used to stain the input image", default='H-DAB')
    parser.add_argument('-p', '--parameters_file', help=".json file containing processing parameters", default=None)
    parser.add_argument('-d', '--debug_level', help="Level of debugging (ex. full, debug, normal, quiet)", default='normal')
    args = parser.parse_args()

    if args.parameters_file:
        params = json.load(open(args.parameters_file))
    else:
        params = {}

    os.makedirs(args.output_path, exist_ok=True)
    
    # preprocess the chunks
    # TODO: move all pdnl_sana.process code into here, then other scripts import these functions
    print('Preprocessing chunks...', flush=True)
    histogram = None
    for d in tqdm(os.listdir(args.input_path)):
        if not os.path.isdir(os.path.join(args.input_path, d)):
            continue
        
        i, j = list(map(int, d.split('_')))

        in_d = os.path.join(args.input_path, f'{i}_{j}')
        out_d = os.path.join(args.output_path, f'{i}_{j}')
        os.makedirs(out_d, exist_ok=True)

        # logger = pdnl_sana.logging.Logger('normal', os.path.join(in_d, 'log.pkl'))
        logger = pdnl_sana.logging.Logger(args.debug_level, os.path.join(in_d, 'log.pkl'))
        level = logger.data['level']
        mpp = logger.data['mpp']
        ds = logger.data['ds']
        converter = pdnl_sana.geo.Converter(mpp, ds)
        frame = pdnl_sana.image.Frame(os.path.join(in_d, 'frame.png'), level=level, converter=converter)
        mask = pdnl_sana.image.Frame(os.path.join(in_d, 'mask.png'), level=level, converter=converter)        

        
        if args.staining_code == 'H-DAB':
            processor = pdnl_sana.process.HDABProcessor(
                logger, frame, main_mask=mask, 
                apply_smoothing=params.get('apply_smoothing', False),
                normalize_background=params.get('normalize_background', False),
                stain_vector=params.get('stain_vector', None)
            )
        if args.staining_code == 'H-DABVR':
            processor = pdnl_sana.process.HDABVRProcessor(
                logger, frame, main_mask=mask, 
                apply_smoothing=params.get('apply_smoothing', False),
                normalize_background=params.get('normalize_background', False),
                stain_vector=params.get('stain_vector', None)
            )
        if args.staining_code == 'CV-DAB':
            processor = pdnl_sana.process.CVDABProcessor(
                logger, frame, main_mask=mask,
                apply_smoothing=params.get('apply_smoothing', False),
                normalize_background=params.get('normalize_background', False),
                stain_vector=params.get('stain_vector', None)
            )                
        if args.staining_code == 'H-DABVR':
            hist = processor.res.get_histogram(mask=processor.main_mask)
        else:
            hist = processor.dab.get_histogram(mask=processor.main_mask)

        if histogram is None:
            histogram = hist
        else:
            histogram = histogram + hist

        if args.staining_code == 'H-DABVR':
            processor.res.save(os.path.join(out_d, 'stain.npy'))
        else:
            processor.dab.save(os.path.join(out_d, 'stain.npy'))

        # if args.debug_level == 'full':
        #     plt.show()
        plt.close()

    # calculate the threshold
    if not histogram is None:
        triangular_strictness = params.get('triangular_strictness', 0.0)
        threshold = pdnl_sana.threshold.triangular_method(
            histogram,
            strictness=triangular_strictness,
            # debug = args.debug_level == 'full'
        )
    else:
        threshold = 0

    minimum_threshold = params.get('minimum_threshold', 0)
    if threshold < minimum_threshold:
        threshold = minimum_threshold
        
    # apply the threshold to the pre-processed chunks
    print('Processing chunks...', flush=True)
    for d in tqdm(os.listdir(args.output_path)):
        if not os.path.isdir(os.path.join(args.input_path, d)):
            continue
        
        i, j = list(map(int, d.split('_')))

        in_d = os.path.join(args.input_path, f'{i}_{j}')
        out_d = os.path.join(args.output_path, f'{i}_{j}')

        mask = pdnl_sana.image.Frame(os.path.join(in_d, 'mask.png'))
        stain = pdnl_sana.image.Frame(os.path.join(out_d, 'stain.npy'))
        
        if args.debug_level == 'full':
            orig = pdnl_sana.image.Frame(os.path.join(in_d, 'frame.png'))
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,8))
            axs = axs.ravel()
            axs[0].imshow(orig.img)
            axs[0].set_title('Original Frame')
            axs[1].imshow(stain.img, cmap='gray')
            axs[1].set_title('Raw Stain')
            axs[2].imshow(mask.img)
            axs[2].set_title('Original Mask')

        stain.mask(mask)
        stain.threshold(threshold)
        morphology_filters = [pdnl_sana.filter.MorphologyFilter(**x) \
                              for x in params.get('morphology_filters', [])]
        for filt in morphology_filters:
            stain.apply_morphology_filter(filt)

        if args.debug_level == 'full':
            axs[3].imshow(stain.img, cmap='gray')
            axs[3].set_title('Processed Stain')
            plt.show()

        stain.save(os.path.join(out_d, 'pos.png'))
        logger = pdnl_sana.logging.Logger('normal', os.path.join(out_d, 'log.pkl'))
        logger.data['triangular_strictness'] = triangular_strictness
        logger.data['minimum_threshold'] = minimum_threshold
        logger.data['morphology_filter'] = morphology_filters
        logger.data['threshold'] = threshold
        logger.write_data()
    
if __name__ == "__main__":
    main()
