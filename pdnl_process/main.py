#!/usr/bin/env python3

import os
import sys
import argparse
import json

import numpy as np
import pdnl_sana.image
import pdnl_sana.process
import pdnl_sana.logging
import pdnl_sana.filter

from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="path to image file to process", required=True)
    parser.add_argument('-o', '--output_path', help="path to write outputs to", required=True)
    parser.add_argument('-m', '--mask_file', help="path to mask file to apply to the image", default=None)
    parser.add_argument('-s', '--staining_code', help="stain protocol used to stain the input image", default='H-DAB')
    parser.add_argument('-p', '--parameters_file', help=".json file containing processing parameters", default=None)
    parser.add_argument('--debug', action='store_true', help="plots an output image to analyze")
    args = parser.parse_args()

    frame = pdnl_sana.image.Frame(args.input_file)
    if args.mask_file:
        mask = pdnl_sana.image.Frame(args.mask_file)
    else:
        mask = None
    if args.parameters_file:
        params = json.load(open(args.parameters_file))
    else:
        params = {}

    file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    debug_level = 'full' if args.debug else 'normal'
    logger = pdnl_sana.logging.Logger(debug_level, os.path.join(args.output_path, 'log.pkl'))

    if args.staining_code == 'H-DAB':
        processor = pdnl_sana.process.HDABProcessor(
            logger, frame, main_mask=mask, 
            apply_smoothing=params.get('apply_smoothing', False),
            normalize_background=params.get('normalize_background', False),
            stain_vector=params.get('stain_vector', None)
        )

    res = processor.run(
        triangular_strictness=params.get('triangular_strictness', 0.0),
        minimum_threshold=params.get('minimum_threshold', 0),
        od_threshold=params.get('od_threshold', None),
        morphology_filters=[pdnl_sana.filter.MorphologyFilter(**filter_params) for filter_params in params.get('morphology_filters', [])],
    )

    if args.debug:
        plt.show()
        

    res['stain'].save(os.path.join(args.output_path, 'stain.npy'))
    res['positive_stain'].save(os.path.join(args.output_path, 'pos.png'))
    logger.write_data()
    
if __name__ == "__main__":
    main()
