# captionqa_eval
Simple eval code for [CaptionQA](https://github.com/bronyayang/CaptionQA).

## Data
Datsets should be organized as follows
```
--data_dir
----captionqa.jsonl
----images
------document
--------images
----------1.jpg
----------2.jpg
------ecommerce
------embodiedai
------natural
```
TODO: add dataset

## Generate caption
```bash
# set api key and url
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=xxx

# set --max_samples to small numbers (e.g. 5) for debug
python -u generate_caption.py --data_dir ../captionqa_dataset \
                             --caption_prompt simple \
                             --model gemini-3.1-pro-preview \
                             --output_path gemini-3.1-pro-preview_results.jsonl \
                             --max_workers 16 \
                             --max_tokens 8192
```

## Evalutation
Here we use `deepseek-chat` for QA models. Origianl paper uses `Qwen/Qwen2.5-72B-Instruct`.
```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1
EVAL_MODEL=deepseek-chat

python -u qa.py --results-path gemini-3.1-pro-preview.jsonl \
                --output-path gemini-3.1-pro-preview-eval-results.json \
                --max-tokens 16 \
                --save-every 100 \
                --num-threads 32 \
                --eval-model $EVAL_MODEL
```
Results per domain and category, and overall summary are reported.
