"""
QA Evaluation with Captions using Borise/CaptionQA Dataset

Evaluates questions using captions instead of images.
Each question is answered once by an LLM using only the caption as context.

Features:
- Uses Qwen2.5-72B-Instruct for evaluation
- Adds "Cannot answer from the caption" option to non-yes/no questions
- Automatic shuffling of answer choices (with order tracking)
- Multi-threaded OpenAI API calls for acceleration

Usage:
    # Evaluate on a specific domain
    python qa.py \
        --caption-path captions.json \
        --output-path results.json \
        --split natural

    # Evaluate on all domains
    python qa.py \
        --caption-path captions.json \
        --output-path results.json \
        --split all
"""

import os
import json
import re
import argparse
import random
import threading
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset
from utils import read_jsonl

LETTER_ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CANNOT_ANSWER_TEXT = "Cannot answer from the caption"
EVAL_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DOMAIN_SPLITS = ["natural", "document", "ecommerce", "embodiedai"]

# Thread-local storage for OpenAI clients
thread_local = threading.local()

# ---------- Helper Functions ----------

def get_openai_client():
    """Get or create thread-local OpenAI client."""
    if not hasattr(thread_local, "client"):
        from openai import OpenAI
        thread_local.client = OpenAI()
    return thread_local.client

def simple_openai_call(client, model, prompt, system_prompt, temperature, max_tokens):
    """Make OpenAI API call."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        n=1,
        max_tokens=max_tokens,
        stream=False
    )
    return response.choices[0].message.content

def process_single_question(args: Tuple) -> Tuple[int, Dict[str, Any]]:
    """
    Process a single question with OpenAI API.
    
    Args:
        args: Tuple containing (idx, prompt, meta_item, system_prompt, eval_model, temperature, max_tokens)
    
    Returns:
        Tuple of (idx, result_dict)
    """
    idx, prompt, meta_item, system_prompt, eval_model, temperature, max_tokens = args
    image_key, q_idx, perm, n_opts, gt_idx_orig, q_data = meta_item
    
    # Get thread-local client
    client = get_openai_client()
    
    # Make API call
    try:
        output = simple_openai_call(
            client, eval_model, prompt, system_prompt, temperature, max_tokens
        )
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        output = ""
    
    # Parse response
    letter = extract_letter(output, n_opts)
    
    is_correct = False
    is_cannot_answer = False
    model_answer_text = None
    score = 0.0
    
    original_choices = q_data["choices"]
    n_original_choices = len(original_choices)
    choices_with_option = add_cannot_answer_option(
        q_data["question"], original_choices
    )
    
    if letter is not None:
        shuf_idx = LETTER_ALPH.find(letter)
        if 0 <= shuf_idx < len(perm):
            orig_idx = perm[shuf_idx]
            
            if orig_idx < len(choices_with_option):
                model_answer_text = str(choices_with_option[orig_idx])
                
                if model_answer_text == CANNOT_ANSWER_TEXT:
                    is_cannot_answer = True
                    score = (1.0 / n_original_choices) + 0.05
                elif orig_idx == gt_idx_orig:
                    is_correct = True
                    score = 1.0
                else:
                    score = 0.0
    
    result_entry = {
        "image_key": image_key,
        "q_idx": q_idx,
        "question": q_data["question"],
        "choices": q_data["choices"],
        "ground_truth": q_data["answer"],
        "model_answer": model_answer_text,
        "model_response": output,
        "is_correct": is_correct,
        "is_cannot_answer": is_cannot_answer,
        "score": round(score, 4),
        "category": q_data.get("category", ""),
        "domain": q_data.get("domain", "")  # Add domain field
    }
    
    return idx, result_entry


def extract_letter(answer_text: str, num_options: int) -> Optional[str]:
    """Extract answer letter from model output."""
    if not answer_text:
        return None

    # If response contains </think>, extract letter from text after it
    if "</think>" in answer_text:
        after_think = answer_text.split("</think>", 1)[1]
        answer_text = after_think

    if "Answer: " in answer_text:
        after_answer = answer_text.split("Answer: ", 1)[1]
        answer_text = after_answer

    if "\n" in answer_text:
        after_n = answer_text.split("\n", 1)[1]
        answer_text = after_n

    m = re.search(r"\b([A-Z])\b", answer_text.upper())
    if m:
        letter = m.group(1)
        idx = LETTER_ALPH.find(letter)
        if 0 <= idx < max(1, num_options):
            return letter
    m = re.search(r"\b([1-9][0-9]?)\b", answer_text)
    if m:
        k = int(m.group(1))
        if 1 <= k <= max(1, num_options):
            return LETTER_ALPH[k - 1]
    return None


def normalize_gt_letter(choices: List[str], answer: str) -> Optional[str]:
    """Extract ground truth answer letter from question."""
    if not choices or not isinstance(answer, str):
        return None

    for i, choice in enumerate(choices):
        if answer.strip() == str(choice).strip():
            return LETTER_ALPH[i]

    return None


def is_yesno_question(question_text: str, choices: List[str]) -> bool:
    """
    Check if question is a yes/no question.
    """
    choice_texts = [str(c).strip().lower() for c in choices]
    has_yes = any("yes" in choice for choice in choice_texts)
    has_no = any("no" in choice for choice in choice_texts)

    if has_yes and has_no:
        return True

    question_lower = question_text.strip().lower()
    yesno_starters = [
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "have ", "has ", "had ", "can ", "could ", "will ", "would ",
        "should ", "shall ", "may ", "might ", "must ",
    ]

    for starter in yesno_starters:
        if question_lower.startswith(starter):
            return True

    return False


def add_cannot_answer_option(question_text: str, choices: List[str]) -> List[str]:
    """Add 'cannot answer from the caption' option to non-yes/no questions."""
    if is_yesno_question(question_text, choices):
        return choices
    return choices + [CANNOT_ANSWER_TEXT]


def build_caption_qa_prompt(caption: str, question: str, choices: List[str]) -> str:
    """Build prompt with caption and question."""
    lines = [f"{LETTER_ALPH[i]}. {choice}" for i, choice in enumerate(choices)]

    prompt = f"""Caption:
{caption}

Question:
{question}

Options:
{chr(10).join(lines)}

Answer:"""

    return prompt


def print_final_summary(results: Dict[str, List], args):
    """Print final evaluation summary with per-category and per-domain breakdown."""
    total_questions = 0
    total_score = 0.0
    correct_answers = 0
    cannot_answer_count = 0

    # Category stats
    category_total = defaultdict(int)
    category_correct = defaultdict(int)
    category_scores = defaultdict(list)
    category_cannot = defaultdict(int)
    
    # Domain stats
    domain_total = defaultdict(int)
    domain_correct = defaultdict(int)
    domain_scores = defaultdict(list)
    domain_cannot = defaultdict(int)

    for v in results.values():
        for item in v:
            # if item is None:
            #     print(v)
            total_questions += 1
            score = item.get("score", 0.0)
            total_score += score
            
            cat = item.get("category", "") or "unknown"
            category_total[cat] += 1
            category_scores[cat].append(score)
            
            domain = item.get("domain", "") or "unknown"
            domain_total[domain] += 1
            domain_scores[domain].append(score)
            
            if item.get("is_correct"):
                correct_answers += 1
                category_correct[cat] += 1
                domain_correct[domain] += 1
                
            if item.get("is_cannot_answer"):
                cannot_answer_count += 1
                category_cannot[cat] += 1
                domain_cannot[domain] += 1

    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    average_score = total_score / total_questions if total_questions > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Model: {args.eval_model}")
    print(f"Threads: {args.num_threads}")

    # Domain results
    if len(domain_total) > 1:
        print(f"\n{'='*60}")
        print(f"Domain Results:")
        print(f"{'='*60}")
        for domain in sorted(domain_total.keys()):
            n = domain_total[domain]
            domain_score = sum(domain_scores[domain]) / n if n else 0.0
            domain_acc = domain_correct[domain] / n if n else 0.0
            domain_cannot_pct = domain_cannot[domain] / n if n else 0.0
            print(f"  {domain}:")
            print(f"    Questions: {n}")
            print(f"    Score: {domain_score:.4f}")
            print(f"    Accuracy: {domain_acc:.2%}")
            print(f"    Cannot answer: {domain_cannot_pct:.2%}")
            print()

    # Category results
    if len(category_total) > 1:
        print(f"\n{'='*60}")
        print(f"Category Results:")
        print(f"{'='*60}")
        for cat in sorted(category_total.keys()):
            n = category_total[cat]
            cat_score = sum(category_scores[cat]) / n if n else 0.0
            cat_acc = category_correct[cat] / n if n else 0.0
            cat_cannot = category_cannot[cat] / n if n else 0.0
            print(f"  {cat}:")
            print(f"    Questions: {n}")
            print(f"    Score: {cat_score:.4f}")
            print(f"    Accuracy: {cat_acc:.2%}")
            print(f"    Cannot answer: {cat_cannot:.2%}")
            print()

    print(f"\n{'='*60}")
    print(f"Overall Summary:")
    print(f"{'='*60}")
    print(f"  Total questions: {total_questions}")
    print(f"  Correct answers: {correct_answers} ({overall_accuracy:.2%})")
    print(f"  'Cannot answer' selections: {cannot_answer_count}")
    print(f"  Total score: {total_score:.2f} / {total_questions}")
    print(f"  Average score: {average_score:.4f}")
    print(f"{'='*60}")


# ---------- Main Evaluation Function ----------

def evaluate_qa_with_captions(args):
    """
    Evaluate questions using captions instead of images.
    Each question is answered once with shuffled choices.
    Multi-threaded implementation for speed.
    """

    # Load dataset
    dataset = read_jsonl(args.results_path)
    captions = {}
    for d in dataset:
        captions[d["id"]] = d["results"]["output"]

    print(f"Loading captions from {args.results_path}...")
    print(f"Using model: {args.eval_model}")
    print(f"Using {args.num_threads} threads for parallel processing")

    # Setup RNG for shuffling
    rng = random.Random(args.seed)

    # Prepare questions
    print("Preparing questions...")
    prompts: List[str] = []
    meta: List[tuple] = []  # (image_id, q_idx, perm, n_opts, gt_idx_orig, original_question_data)

    skipped_no_caption = 0
    skipped_no_choices = 0

    for entry in dataset:
        image_id = entry.get("id")
        if image_id is None:
            continue

        image_key = str(image_id)
        if image_key not in captions:
            skipped_no_caption += 1
            continue
        caption = captions[image_key]

        # Get domain from entry
        domain = entry.get("domain", "")

        questions = entry.get("questions", [])
        if not questions:
            if "question" in entry:
                cat = entry.get("category", [])
                if isinstance(cat, list):
                    cat = cat[0] if cat else ""
                questions = [
                    {
                        "question": entry["question"],
                        "choices": entry.get("choices", []),
                        "answer": entry.get("answer"),
                        "category": cat,
                        "domain": domain,
                    }
                ]
            else:
                continue

        for q_idx, q in enumerate(questions):
            question_text = q.get("question", "")
            choices = q.get("choices", [])
            answer = q.get("answer")
            
            category = q.get("category", [])
            if isinstance(category, list):
                category = category[0] if category else ""

            if not choices or len(choices) < 2:
                skipped_no_choices += 1
                continue

            gt_letter_orig = normalize_gt_letter(choices, answer)
            if gt_letter_orig is None:
                continue
            gt_idx_orig = LETTER_ALPH.index(gt_letter_orig)

            choices_with_option = add_cannot_answer_option(question_text, choices)

            n_opts = len(choices_with_option)
            perm = list(range(n_opts))
            rng.shuffle(perm)

            shuffled_opts = [choices_with_option[i] for i in perm]
            prompt = build_caption_qa_prompt(caption, question_text, shuffled_opts)

            # Add domain to question data
            q_data = {
                "question": question_text,
                "choices": choices,
                "answer": answer,
                "category": category,
                "domain": domain,
            }

            prompts.append(prompt)
            meta.append((image_key, q_idx, perm, n_opts, gt_idx_orig, q_data))

    print(f"Prepared {len(prompts)} questions")
    print(f"Skipped: {skipped_no_caption} (no caption), {skipped_no_choices} (no choices)")

    if not prompts:
        print("No questions to evaluate!")
        return

    # System prompt
    system_prompt = "You are given a caption describing an image, and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation."

    # Load existing results if present
    results = {}
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    try:
        with open(args.output_path, "r", encoding="utf-8") as f:
            results = json.load(f) or {}
        print(f"Loaded existing results from {args.output_path} (resume mode)")
    except Exception as e:
        print(f"Starting fresh - no existing results found")

    # Map image -> already processed count
    processed_count = {k: len(v) for k, v in results.items() if isinstance(v, list)}

    # Determine which questions still need processing
    indices_to_process = []
    task_args = []
    
    for i, (image_key, q_idx, _perm, _n_opts, _gt_idx_orig, _q_data) in enumerate(meta):
        done = processed_count.get(image_key, 0)
        if q_idx >= done:
            indices_to_process.append(i)
            task_args.append((
                i, prompts[i], meta[i], system_prompt, 
                args.eval_model, args.temperature, args.max_tokens
            ))

    total_remaining = len(indices_to_process)
    print(f"Already processed: {len(prompts) - total_remaining}; remaining: {total_remaining}")

    if total_remaining == 0:
        print_final_summary(results, args)
        return

    # Process questions in parallel
    print(f"Starting parallel processing with {args.num_threads} threads...")
    
    results_lock = threading.Lock()
    processed_in_session = 0
    total_score_in_session = 0.0
    correct_in_session = 0
    cannot_in_session = 0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_question, arg): arg[0] 
            for arg in task_args
        }
        
        # Process results as they complete
        with tqdm(total=total_remaining, desc="Processing questions") as pbar:
            for future in as_completed(future_to_idx):
                idx, result_entry = future.result()
                
                image_key = result_entry["image_key"]
                q_idx = result_entry["q_idx"]
                
                # Remove temporary fields
                del result_entry["image_key"]
                del result_entry["q_idx"]
                
                # Update results with lock
                with results_lock:
                    if image_key not in results:
                        results[image_key] = []
                    
                    # Ensure correct position
                    while len(results[image_key]) <= q_idx:
                        results[image_key].append(None)
                    results[image_key][q_idx] = result_entry
                    
                    # Update session stats
                    processed_in_session += 1
                    total_score_in_session += result_entry["score"]
                    if result_entry["is_correct"]:
                        correct_in_session += 1
                    if result_entry["is_cannot_answer"]:
                        cannot_in_session += 1
                
                # Save periodically
                if processed_in_session % args.save_every == 0:
                    with results_lock:
                        with open(args.output_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=4, ensure_ascii=False)
                        
                        # Print progress
                        session_avg = total_score_in_session / processed_in_session
                        session_acc = correct_in_session / processed_in_session
                        print(f"\n[Progress - Session] Processed: {processed_in_session}/{total_remaining}")
                        print(f"  Avg score: {session_avg:.4f}, Accuracy: {session_acc:.2%}")
                        print(f"  Cannot answer: {cannot_in_session}")
                
                pbar.update(1)
    
    # Final save
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nAll processing complete. Results saved to {args.output_path}")
    
    # Print final summary
    print_final_summary(results, args)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate questions using captions with Borise/CaptionQA dataset"
    )
    parser.add_argument("--results-path", type=str, default="debug.jsonl")
    parser.add_argument("--output-path", type=str, required=True, 
                       help="Path to save evaluation results")
    parser.add_argument("--max-tokens", type=int, default=4,
                       help="Maximum tokens for response (default: 4)")
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for option shuffling (default: 0)")
    parser.add_argument("--save-every", type=int, default=50,
                       help="Save incremental results every N questions (default: 50)")
    parser.add_argument("--eval-model", type=str, default=EVAL_MODEL,
                       help=f"model for QA (default: {EVAL_MODEL})")
    parser.add_argument("--num-threads", type=int, default=4,
                       help="Number of threads for parallel API calls (default: 4)")
    
    args = parser.parse_args()

    args_dict = vars(args)
    print("=====Args=====")
    for key, value in args_dict.items():
        print(f"  {key}: {value}")
    print("======> Starting evaluation")
    
    evaluate_qa_with_captions(args)


if __name__ == "__main__":
    main()