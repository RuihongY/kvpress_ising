dataset="ruler"
data_dir="4096"
model="Qwen/Qwen2-7B-Instruct"
compression_ratio=0.8
fraction=1.0
press_names=("snapkv")

# QUBO parameters for SnapKVPress
qubo_ratios=(0.0 0.1 0.2)
similarity_threshold=0.85
lambda_penalty=2.0
lagrange_multiplier=10.0
qubo_solver_method="parallel_annealing"
qubo_num_iterations=100
qubo_initial_temp=10.0
qubo_cooling_rate=0.95
qubo_num_chains=20

# Check if the number of press names is less than or equal to the number of available GPUs
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ ${#press_names[@]} -gt $num_gpus ]; then
  echo "Error: The number of press names (${#press_names[@]}) exceeds the number of available GPUs ($num_gpus)"
  exit 1
fi

# Iterate over press names and qubo ratios
for i in "${!press_names[@]}"; do
  press="${press_names[$i]}"
  
  # Run each press_name on a different GPU in the background
  (
    for qubo_ratio in "${qubo_ratios[@]}"; do
      echo "Running press_name: $press with compression_ratio: $compression_ratio, qubo_ratio: $qubo_ratio on GPU cuda:$i"
      python evaluate.py \
        --dataset $dataset \
        --data_dir $data_dir \
        --model $model \
        --press_name $press \
        --compression_ratio $compression_ratio \
        --fraction $fraction \
        --qubo_ratio $qubo_ratio \
        --similarity_threshold $similarity_threshold \
        --lambda_penalty $lambda_penalty \
        --lagrange_multiplier $lagrange_multiplier \
        --qubo_solver_method $qubo_solver_method \
        --qubo_num_iterations $qubo_num_iterations \
        --qubo_initial_temp $qubo_initial_temp \
        --qubo_cooling_rate $qubo_cooling_rate \
        --qubo_num_chains $qubo_num_chains \
        --device "cuda:$i"
    done
  ) &
done

# Wait for all background jobs to finish
wait
echo "All evaluations completed."