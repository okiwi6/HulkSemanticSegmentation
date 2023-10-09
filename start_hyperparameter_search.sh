. .venv/bin/activate

STUDY_NAME="SemanticSegmentation_4classes_downscale4_focal_lion"
STORAGE="optuna_semantic_segmentation"

echo $(which python)
echo "Running study: ${STUDY_NAME}"
CUDA_VISIBLE_DEVICES=0 nohup python -m src.hyperparameter_search ${STUDY_NAME} ${STORAGE} 0 > log_gpu0 2>&1 &
# Required if the database is created by first process
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup python -m src.hyperparameter_search ${STUDY_NAME} ${STORAGE} 0 > log_gpu1 2>&1 &
# nohup python -m src.hyperparameter_search ${STUDY_NAME} ${STORAGE} > log 2>&1 &
