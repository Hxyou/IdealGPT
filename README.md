# IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models.

## Run
To support LLaVA or MiniGPT4 as the model to solve sub-question, need to install llava/minigpt4 as mentioned in their repo.

```Shell
python blip_gpt_main.py  \
    --data_root=/home/haoxuan/data/vcr1/ \
    --exp_tag=vcr_05241522 \
    --dataset=vcr_val \
    --device_id=0 \
    --prompt_setting=v1a \
    --data_partition=0_499 \
    --vqa_model=blip2_t5_xl  \
    --temp_gpt=0.0  \
    --data_subset=/home/haoxuan/code/SubGPT/exp_data/vcr_val_random500_annoid.yaml \
    --openai_key=xxx
```

## Eval

```
python vcr_eval.py --result=[path of saved VCR result folder, named by exp_tag in run]
```

