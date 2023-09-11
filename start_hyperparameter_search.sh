. .venv/bin/activate

STUDY_NAME="semseg_4c_1"

echo $(which python)
echo "Running study: ${STUDY_NAME}"
CUDA_VISIBLE_DEVICES=0 nohup python -m src.hyperparameter_search ${STUDY_NAME} > log_gpu0 2>&1 &
# Required if the database is created by first process
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup python -m src.hyperparameter_search ${STUDY_NAME} > log_gpu1 2>&1 &
