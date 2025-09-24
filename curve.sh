#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

num_clients_values=(2 4 6 8 10 12 14)
num_time_slots_values=(11 13)
alg_name_values=("PMET")
for alg_name in "${alg_name_values[@]}"; do
    for num_time_slots in "${num_time_slots_values[@]}"; do
        echo "Running with alg_name=${alg_name} and num_time_slots=${num_time_slots} dataset = zsre"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=zsre  --generation_test_interval=10 --num_time_slots=$num_time_slots --num_clients=1 
        echo "Running with alg_name=${alg_name}-CKE and num_time_slots=${num_time_slots} dataset = zsre"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=zsre  --generation_test_interval=10 --num_time_slots=$num_time_slots
        echo "Running with alg_name=${alg_name}-CPKE and num_time_slots=${num_time_slots} dataset = zsre"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=zsre  --generation_test_interval=10 --num_time_slots=$num_time_slots --personalize --similarity  0.6

        echo "Running with alg_name=${alg_name} and num_time_slots=${num_time_slots} dataset = mcf"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=mcf --generation_test_interval=10 --num_time_slots=$num_time_slots --num_clients=1 
        echo "Running with alg_name=${alg_name}-CKE and num_time_slots=${num_time_slots} dataset = mcf"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=mcf  --generation_test_interval=10 --num_time_slots=$num_time_slots
        echo "Running with alg_name=${alg_name}-CPKE and num_time_slots=${num_time_slots} dataset = mcf"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=mcf  --generation_test_interval=10 --num_time_slots=$num_time_slots --personalize --similarity  0.6
    done
done

alg_name_values=("PMET")
for alg_name in "${alg_name_values[@]}"; do
    for num_clients in "${num_clients_values[@]}"; do
        echo "Running with alg_name=${alg_name} and num_clients=${num_clients} dataset = zsre"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=zsre  --generation_test_interval=10 --num_clients=1 
        echo "Running with alg_name=${alg_name}-CKE and num_clients=${num_clients} dataset = zsre"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=zsre  --generation_test_interval=10 --num_clients=$num_clients
        echo "Running with alg_name=${alg_name}-CPKE and num_clients=${num_clients} dataset = zsre"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=zsre  --generation_test_interval=10 --num_clients=$num_clients --personalize --similarity  0.6

        echo "Running with alg_name=${alg_name} and num_clients=${num_clients} dataset = mcf"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=mcf  --generation_test_interval=10 --num_clients=1 
        echo "Running with alg_name=${alg_name}-CKE and num_clients=${num_clients} dataset = mcf"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=mcf  --generation_test_interval=10 --num_clients=$num_clients
        echo "Running with alg_name=${alg_name}-CPKE and num_clients=${num_clients} dataset = mcf"
        python -m fededit.evaluate --alg_name=$alg_name --model_name=EleutherAI/gpt-j-6B --hparams_fname=EleutherAI_gpt-j-6B.json --ds_name=mcf  --generation_test_interval=10 --num_clients=$num_clients --personalize --similarity  0.6
    done
done

echo "All commands executed."

exit 0


# run_013 start