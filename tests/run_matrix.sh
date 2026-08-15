#!/bin/bash
# 并发与长上下文压测矩阵（服务已开 256k + MTP x3）
cd /home/dgx/projects/qwen38-test
V=/home/dgx/projects/qwen38-env/bin/vllm
ARGS="--model unsloth/Qwen3.8-27B-NVFP4 --tokenizer /home/dgx/projects/models/Qwen3.8-27B-NVFP4 --base-url http://127.0.0.1:8000 --dataset-name random --ignore-eos"
OUT=bench_matrix.txt
: > $OUT

run() { # $1=标题 $2=input_len $3=output_len $4=num_prompts $5=concurrency
  echo "########## $1 | in=$2 out=$3 n=$4 c=$5 ##########" >> $OUT
  $V bench serve $ARGS --random-input-len $2 --random-output-len $3 \
    --num-prompts $4 --max-concurrency $5 2>&1 \
    | sed -n '/Serving Benchmark Result/,$p' >> $OUT
  echo >> $OUT
}

echo "=== [1/6] 并发1 ==="
run "并发1" 512 256 2 1
echo "=== [2/6] 并发2 ==="
run "并发2" 512 256 4 2
echo "=== [3/6] 并发4 ==="
run "并发4" 512 256 8 4
echo "=== [4/6] 上下文1k ==="
run "上下文1k" 1024 128 2 1
echo "=== [5/6] 上下文100k（约2.5分钟）==="
run "上下文100k" 102400 128 2 1
echo "=== [6/6] 上下文200k（约4分钟）==="
run "上下文200k" 204800 128 2 1
echo "ALL DONE"
