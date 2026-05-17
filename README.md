# CaptionQA Eval

Simple evaluation code for [CaptionQA](https://github.com/bronyayang/CaptionQA).

## Data

Datasets should be organized as follows:

```text
data_dir/
├── captionqa.jsonl
└── images/
    ├── document/
    │   └── images/
    │       ├── 1.jpg
    │       └── 2.jpg
    ├── ecommerce/
    ├── embodiedai/
    └── natural/
```

TODO: Add dataset download instructions.

---

## Generate Captions

```bash
# Set API key and base URL
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=xxx

# Set --max_samples to a small number (e.g. 5) for debugging
python -u generate_caption.py \
    --data_dir ../captionqa_dataset \
    --caption_prompt simple \
    --model gemini-3.1-pro-preview \
    --output_path gemini-3.1-pro-preview_results.jsonl \
    --max_workers 16 \
    --max_tokens 8192
```
Currently, only OpenAI style api is supported. If you want to test local models, please use `vLLM` or `SGLang` to serve the model.

TODO: support local inference with `vLLM` and `transformers` backend.

---

## Evaluation

Here we use `deepseek-chat` as the QA evaluation model.  
The original paper uses `Qwen/Qwen2.5-72B-Instruct`.

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1

EVAL_MODEL=deepseek-chat

python -u qa.py \
    --results-path gemini-3.1-pro-preview_results.jsonl \
    --output-path gemini-3.1-pro-preview-eval-results.json \
    --max-tokens 16 \
    --save-every 100 \
    --num-threads 32 \
    --eval-model $EVAL_MODEL
```

Results are reported per domain, per category, and as an overall summary.
