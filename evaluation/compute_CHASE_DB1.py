import argparse
import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import pandas as pd
from skimage import segmentation, io, measure
import tifffile as tif
import os
join = os.path.join
from tqdm import tqdm
import traceback
from evaluation.metrics import Evaluator

parser = argparse.ArgumentParser('Compute F1 score for cell segmentation results', add_help=False)
# Dataset parameters
parser.add_argument('-g', '--gt_path', default='your_address/nnUNet_data/nnUNet_raw/Dataset270_Vessel/labelsTs', type=str, help='path to ground truth')
parser.add_argument('-s', '--seg_path', type=str, default='your_address', help='path to segmentation results; file names are the same as ground truth', required=False)
#parser.add_argument('-thre', '--thresholds', nargs='+', default=[0.5], type=float, help='threshold to count correct cells')
#parser.add_argument('-o', '--output_path', default='', type=str, help='path where to save metrics')
#parser.add_argument('--count_bd_cells', default=False, action='store_true', required=False, help='remove the boundary cells when computing metrics by default')
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
evaluator = Evaluator(2)
names = sorted(os.listdir(seg_path))
# names = [i for i in names if i.endswith('.png')]
# names = [i for i in names if os.path.isfile(join(gt_path, i.split('.png')[0]+'_label.tiff'))]

print('num of files:', len(names)-3)
for name in tqdm(names):
    if name =='dataset.json' or name=='plans.json' or name=='predict_from_raw_data_args.json':
        continue
    gt = io.imread(join(gt_path, name), as_gray=True)
    seg = io.imread(join(seg_path, name))
    gt[gt > 0] = 1
    gt[gt<= 0] = 0
    seg[seg>=0.5] = 1
    seg[seg<0.5] = 0
    evaluator.add_batch(gt, seg)
mIoU = evaluator.Mean_Intersection_over_Union()
IoU = evaluator.Intersection_over_Union()
F1 = evaluator.F1_Score()
Sensitivity = evaluator.Sensitivity()

print(mIoU,IoU, F1, Sensitivity)

