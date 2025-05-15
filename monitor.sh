while true; do
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")
  disk_usage=$(du -sh activations/ | cut -f1)
  gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{print $1"% GPU, "$2"/"$3" MB"}')
  echo "$timestamp | Disk: $disk_usage | GPU: $gpu_usage"
  sleep 5
done