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
    parser.add_argument('--debug_level', help="outputs extra files", default='normal')
    args = parser.parse_args()

    if args.parameters_file:
        params = json.load(open(args.parameters_file))
    else:
        params = {}

    os.makedirs(args.output_path, exist_ok=True)

    # get the background image
    # TODO: test how big the concatenated frame will be, and move to big roi mode if too big
    rois = pdnl_sana.slide.concatenate_chunks(args.input_path)
    for roi_name in rois:
        frame, mask, frame_loc = rois[roi_name]    

        # TODO: move all pdnl_sana.process code into here, then other scripts import these functions
        out_d = os.path.join(args.output_path, roi_name)
        os.makedirs(out_d, exist_ok=True)
        logger = pdnl_sana.logging.Logger(args.debug_level, os.path.join(out_d, 'log.pkl'))

        # TODO: shouldn't be omitting the mask here, but leaving it here so all data is generated in the same way for now.
        #       this was a fix for the background code, but that code is solved now
        if args.staining_code == 'H-DAB':
            processor = pdnl_sana.process.HDABProcessor(
                logger, frame, 
                #main_mask=mask, 
                apply_smoothing=params.get('apply_smoothing', False),
                normalize_background=params.get('normalize_background', False),
                stain_vector=params.get('stain_vector', None),
                radius=200, overlap=0.5,
            )
        if args.staining_code == 'CV-DAB':
            processor = pdnl_sana.process.CVDABProcessor(
                logger, frame, 
                #main_mask=mask,
                apply_smoothing=params.get('apply_smoothing', False),
                normalize_background=params.get('normalize_background', False),
                stain_vector=params.get('stain_vector', None)
            )                
        hist = processor.dab.get_histogram(mask=processor.main_mask)
        stain = processor.dab.copy()
        stain.save(os.path.join(out_d, 'stain.npy'))

        # calculate the threshold
        triangular_strictness = params.get('triangular_strictness', 0.0)
        threshold = pdnl_sana.threshold.triangular_method(
            hist,
            strictness=triangular_strictness,
        )

        minimum_threshold = params.get('minimum_threshold', 0)
        if threshold < minimum_threshold:
            threshold = minimum_threshold

        stain.mask(mask)
        stain.threshold(threshold)
        morphology_filters = [pdnl_sana.filter.MorphologyFilter(**x) \
                              for x in params.get('morphology_filters', [])]
        for filt in morphology_filters:
            stain.apply_morphology_filter(filt)

        if args.debug_level in ['debug', 'full']:
            frame.save(os.path.join(out_d, 'frame.png'))   

        if args.debug_level == 'full':
            plt.show()     
        stain.save(os.path.join(out_d, 'pos.png'))
        mask.save(os.path.join(out_d, 'mask.png'))
        logger.data['triangular_strictness'] = triangular_strictness
        logger.data['minimum_threshold'] = minimum_threshold
        logger.data['morphology_filter'] = morphology_filters
        logger.data['threshold'] = threshold
        logger.data['frame_loc'] = frame_loc
        logger.write_data()
    
if __name__ == "__main__":
    main()
