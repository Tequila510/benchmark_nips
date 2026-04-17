# Evaluation Instructions

1. run AppAgent-v1
```shell
cd <your_path>/evaluation/agents/AppAgent
python ./scripts/task_executor_offline.py --dataset ../../../data/douyin_20/checkpoint_douyin.json --output ../../results/appagent-v1.json
```
2. run Mobile-Agent-v2
```shell
cd <your_path>/evaluation/agents/UI-TARS
python ./scripts/task_executor_offline.py --dataset ../../../data/douyin_20/checkpoint_douyin.json --output ../../results/ui-tars.json
``` 
2. run UI-TARS
