import argparse
import copy
import json
import os
import threading
import time
import traceback
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict

from PIL import Image
from openai import OpenAI
from tqdm import tqdm

from utils import read_json, write_json, read_jsonl, write_jsonl

file_lock = threading.Lock()
uni_id = "id"
CAPTION_PROMPTS = {
    "simple": "Describe this image in detail.",
    "short": "Write a very short caption for the given image.",
    "long": "Write a very long and detailed caption describing the given image as comprehensively as possible.",
}


def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            byte_data = image_file.read()
    else:
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def get_single_data_results(payload, client):
    response = client.chat.completions.create(**payload)
    choices = response.choices
    if choices:
        output = response.choices[0].message.content
    else:
        output = "EMPTY"
    model = response.model
    usage = response.usage
    raw_response = response.to_dict()
    return {"output":output,"raw_response":raw_response}


def process_single_data(client, args, data):
    prompt = CAPTION_PROMPTS[args.caption_prompt]
    image_paths = data["image_paths"]
    domain = data["domain"]
    image_paths = [f"{args.data_dir}/images/{p}" for p in image_paths]
    encoded_images = [encode_image(img_path) for img_path in image_paths]
    content_items = []
    for encoded_image in encoded_images:
        content_items.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        )

    messages = [
        {
            "role": "user",
            "content": content_items,
        },
        {"role": "user", "content": prompt},
    ]

    payload = {}
    payload["model"] = args.model
    payload["messages"] = messages
    payload["temperature"] = args.temperature
    payload["top_p"] = args.top_p
    payload["max_tokens"] = args.max_tokens
    payload["n"] = args.n
    # most models use thinking as default, so use --disable_thinking
    # if you do not set --reasoning-parser in vllm,output will include ...</think>
    # set --reasoning-parser,get reasoning content and tokens, but for caption, it is ok for all.
    
    extra_args = {}
    extra_args["top_k"] = args.top_k
    if args.disable_thinking:
        extra_args.update({"chat_template_kwargs": {"enable_thinking": False}})
        # payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    ################
    # TODO, extra ?
    if args.presence_penalty != 1.0:
        payload["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty != 1.0:
        payload["repetition_penalty"] = args.repetition_penalty
    ############

    payload["extra_body"] = extra_args
    result = copy.deepcopy(data)
    response = get_single_data_results(payload, client)
    result["results"] = response

    with file_lock:
        with open(args.output_path, "a", encoding="utf-8") as g:
            g.write(json.dumps(result, ensure_ascii=False) + "\n")
            g.flush()

    return result


def resume_data(data_list, args):
    if os.path.exists(args.output_path):
        finish_data = []
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                item = json.loads(line)
                finish_data.append(item)
        finish_data_id = [d[uni_id] for d in finish_data]
        remain_data = []
        for d in data_list:
            if d[uni_id] in finish_data_id:
                continue
            remain_data.append(d)
        print("The number of raw examples", len(data_list))
        print("## Finish examples:", len(finish_data_id))
        print("## Remain examples:", len(remain_data))
        return remain_data
    else:
        print(f"{args.output_path} does not exist, no finished examples.")
        return data_list


def validate_output_path(output_path):
    if not output_path.endswith(".jsonl"):
        raise ValueError(f"Output path must be a .jsonl file. Got: {output_path}")

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../captionqa_dataset", type=str)
    parser.add_argument(
        "--caption_prompt",
        default="simple",
        type=str,
        choices=list(CAPTION_PROMPTS.keys()),
    )
    parser.add_argument("--model", default="gpt-4o", type=str)
    parser.add_argument(
        "--output_path",
        type=str,
        default="results.jsonl",
        help="please use jsonl files",
    )
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--presence_penalty",default=1.0)
    parser.add_argument("--repetition_penalty",default=1.0)
    parser.add_argument(
    "--disable_thinking",
    action="store_true",
    help="Disable thinking mode"
)
    args = parser.parse_args()

    # Validate output path
    try:
        validate_output_path(args.output_path)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    print("Args:")
    for k, v in vars(args).items():
        print(f"{k}:{v}")
    return args


def main():
    args = get_args()
    jsonl_path = f"{args.data_dir}/captionqa.jsonl"
    text_data = read_jsonl(jsonl_path)

    if uni_id not in list(text_data[0].keys()):
        print(f"Add unique id: <{uni_id}> for all samples")
        res = []
        for idx, data in enumerate(text_data):
            d = copy.deepcopy(data)
            d[uni_id] = idx
            res.append(d)
        text_data = res

    client = OpenAI()
    # model_name = client.models.list().data[0].id  # check model list
    # print(model_name)
    # assert 0
    
    if args.max_samples > 0:
        text_data = text_data[: args.max_samples]
        print(f"Only run {args.max_samples} samples")

    completed_count = 0
    text_data = resume_data(text_data, args)
    total_count = len(text_data)
    print("Num test samples:", total_count)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_data, client, args, data): data
            for data in text_data
        }

        for future in tqdm(as_completed(futures), total=total_count, desc="Processing"):
            try:
                response = future.result()
                completed_count += 1
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

    end_time = time.time()
    cost_time = (end_time - start_time) / 3600
    print(f"Completed {completed_count}/{total_count} samples")
    print(f"Results saved to {args.output_path}, cost time {cost_time:.2f} hours.")


if __name__ == "__main__":
    main()
