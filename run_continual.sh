# _except_mvtec_visa
# _except_continual_ad

JSON_PATH_BASE="base_classes_except_continual_ad"
TASK_JSON_GROUPS=(
    "5classes_tasks_except_continual_ad" \
    "10classes_tasks_except_continual_ad" \
    "30classes_tasks_except_continual_ad"
)
NUM_TASKS_GROUPS=(
    6 \
    3 \
    1
)

echo "[INFO] Start BASE PHASE"
python main.py \
    --gpu 0 \
    --seed 0 \
    --log_group simplenet_continual \
    --log_project ContinualAD \
    --results_path results \
    --run_name base_model \
    --json_path $JSON_PATH_BASE \
    --task_id 0 \
    net \
    --meta_epochs 50 \
    dataset \
    --batch_size 16 \
    --num_workers 8 \
    --json_path $JSON_PATH_BASE \
    --task_id 0
    
# CONTINUAL 학습 (task_id=1~5)
for ((i=0; i<${#TASK_JSON_GROUPS[@]}; i++)); do
    NUM_TASKS=${NUM_TASKS_GROUPS[$i]}
    for ((TASK_ID=1; TASK_ID<=NUM_TASKS; TASK_ID++)); do
        echo "[INFO] Start CONTINUAL PHASE - Task $TASK_ID"
        python main.py \
            --gpu 0 \
            --seed 0 \
            --log_group simplenet_continual \
            --log_project ContinualAD \
            --results_path results \
            --run_name base_model \
            --json_path "${TASK_JSON_GROUPS[i]}" \
            --task_id $TASK_ID \
            net \
            --meta_epochs 20 \
            dataset \
            --batch_size 16 \
            --num_workers 8 \
            --json_path "${TASK_JSON_GROUPS[i]}" \
            --task_id $TASK_ID
    done
done
