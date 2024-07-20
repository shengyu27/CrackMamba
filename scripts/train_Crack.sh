
MAMBA_MODEL='nnUNetTrainerCrackMamba'
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset227_Crack/${MAMBA_MODEL}/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset227_Crack/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="1"

# train
#CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 227 2d all -tr ${MAMBA_MODEL} &&

# predict
echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "your_address/nnUNet_data/nnUNet_raw/Dataset227_Crack/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 227 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \ nnUNetTrainerSwinUMamba
    --disable_tta \
    -f all \
    -chk "your_address/checkpoint_best.pth" &&

echo "Computing F1 and mIou..."
python evaluation/compute_CrackSeg9k.py \
    --gt_path "your_address/nnUNet_data/nnUNet_raw/Dataset227_Crack/labelsTs" \
    -s "${PRED_OUTPUT_PATH}"  &&

echo "Done."