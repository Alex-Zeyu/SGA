#!/bin/bash
source ../myenv/bin/activate

# Define the parameter settings
# augment_values=("True" "False")
# train_plan_values=("True" "False")
# model_values=("gcn" "gat")
# dataset_values=("bitcoin-alpha" "Epinions" "bitcoin-otc" "Slashdot" "wiki-elec" "wiki-RfA")

# Loop through each combination of parameter settings
# for augment in "${augment_values[@]}"; do
#     for train_plan in "${train_plan_values[@]}"; do
#         for model in "${model_values[@]}"; do
#             for dataset in "${dataset_values[@]}"; do
#                 # Execute unsigned_baseline3.py with the current parameter settings using nohup to avoid stopping after disconnecting from remote
#                 nohup python unsigned_baseline3.py --augment "$augment" --train_plan "$train_plan" --model "$model" --dataset "$dataset" >> result.txt
#             done
#         done
#     done
# done


# for model in "${model_values[@]}"; do
#     for dataset in "${dataset_values[@]}"; do
#         # Execute unsigned_baseline3.py with the current parameter settings using nohup to avoid stopping after disconnecting from remote
#         nohup python unsigned_baseline3.py --model "$model" --dataset "$dataset" --train_plan=False >> result20240513.txt
#     done
# done

params_set=(
    "gat Epinions 1 1"
    "gat Slashdot 1 1"
    "gat wiki-RfA 1 1"
    "gcn Slashdot 1 1"
)

for params in "${params_set[@]}"; do
    set -- $params
    nohup python unsigned_baseline.py --model $1 --dataset $2 --augment $3 --train_plan $4 >> unsigned_baseline.log
done

deactivate
