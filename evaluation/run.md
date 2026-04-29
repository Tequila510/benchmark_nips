# RUN AGENTS Instructions

1. run AppAgent-v1

- Add `apiKey` and `apiBaseUrl` in `evaluation/agents/AppAgent/config.yaml`:

- run task_executor_offline.py
```shell
cd evaluation/agents/AppAgent
conda create -n AppAgent-v1 python=3.12.13
conda activate AppAgent-v1
pip install -r requirements_offline.txt
python ./scripts/task_executor_offline.py \
    --dataset ../../../data/hupu-15/checkpoint_hupu.json \
    --output ../../results/appagent-v1_glm5.json
```

2. run Mobile-Agent-v2 or Mobile-Agent-E
```shell
cd evaluation/agents/MobileAgent/Mobile-Agent-v2
conda create -n Mobile-Agent-v2 python=3.10.20
conda activate Mobile-Agent-v2
pip install -r requirements_offline.txt
python run_offline.py \
    --dataset ../../../../data/taobao-16/checkpoint_taobao.json \
    --output ../../../results/mobile-agent-v2_gpt4o.json \
    --api_url "" \
    --api_key "" \
    --model gpt-4o \
    --qwen_api "" \
    --caption_model qwen-vl-max \
    --use_reflection --use_memory
    
cd evaluation/agents/MobileAgent/Mobile-Agent-E
conda create -n Mobile-Agent-E python=3.10.20
conda activate Mobile-Agent-E
pip install -r requirements_offline.txt
python run_offline.py \
    --dataset ../../../../data/douyin-20/checkpoint_douyin.json \
    --output ../../../results/mobile-agent-e_gpt5.json \
    --api_url "" \
    --api_key "" \
    --model gpt-5 \
    --qwen_api "" \
    --caption_model qwen-vl-max \
    --enable_evolution --enable_experience_retriever
```
3. run ClawMobile

```shell
conda create -n ClawMobile python=3.12
conda activate ClawMobile
pip install -r requirements_offline.txt

uv run python run_offline_mobilerun.py \
    --dataset ../../../data/kugou-15/checkpoint_kugou.json \
    --output ../../results/clawmobile_gpt4o.json \
    --api_key "" \
    --api_base "" \
    --model gpt-4o \
    --reasoning

```

4. run Fine-tuned Models
```shell
cd evaluation/agents/FineTunedModel
conda create -n finetunedmodel python=3.12.13
conda activate finetunedmodel
pip install -r requirements_offline.txt

python run.py \
--dataset "../../../data/bilibili-15/checkpoint_bilibili.json" \
--output "../../results/UI-TARS-7B.json" \
--model "ui-tars-1.5-7b"

python run_gui.py \ 
--dataset "../../../data/hupu-15/checkpoint_hupu.json" \
--output "../../results/GUI-Owl-7B.json" \
--model gui-owl-7b

```


# EVALUATION instructions

```shell
cd <your_path>/benchmark_nips/evaluation
python metrics.py <agent_result_json_path>
python metrics.py ./results/mobile-agent-v2_gpt5.json

```