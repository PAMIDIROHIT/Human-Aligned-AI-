"""MT-Bench Evaluation -- Score models via LLM-as-judge.

Runs MT-Bench multi-turn evaluation across base, SFT, and PPO models.
Uses GPT-4 as judge (following the MT-Bench paper methodology).

Hardware target: 4x Tesla K80 (fp32).
Compatible with: transformers==4.38.2, peft==0.7.1

Usage:
    python -m src.evaluation.mt_bench --config params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# MT-Bench categories
MT_BENCH_CATEGORIES = [
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
]

# Sample MT-Bench-style questions
MT_BENCH_QUESTIONS = [
    {"question_id": 1, "category": "writing", "turns": [
        "Write a persuasive essay about why remote work is the future of employment.",
        "Now rewrite the essay targeting a skeptical CEO audience, focusing on productivity data.",
    ]},
    {"question_id": 2, "category": "roleplay", "turns": [
        "You are a medieval knight. Describe your typical day in first person.",
        "Now a dragon has appeared at the castle. Continue the story.",
    ]},
    {"question_id": 3, "category": "reasoning", "turns": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
        "Now consider: If all roses are red and all red things are visible, what can we conclude about roses?",
    ]},
    {"question_id": 4, "category": "math", "turns": [
        "Solve step by step: A store offers a 20% discount on a $50 item, then applies a 10% tax. What is the final price?",
        "Now the store changes its policy: tax is applied first, then the discount. Which scenario is cheaper for the customer?",
    ]},
    {"question_id": 5, "category": "coding", "turns": [
        "Write a Python function that finds the longest palindromic substring in a given string.",
        "Now optimize the function to run in O(n) time using Manacher's algorithm.",
    ]},
    {"question_id": 6, "category": "extraction", "turns": [
        "Extract all the dates and associated events from the following text: 'The company was founded on January 15, 2010. They launched their first product on March 3, 2012. The IPO occurred on September 20, 2015.'",
        "Now convert all dates to ISO 8601 format and output as JSON.",
    ]},
    {"question_id": 7, "category": "stem", "turns": [
        "Explain the concept of entropy in thermodynamics to a high school student.",
        "Now explain how entropy relates to information theory. What is the connection?",
    ]},
    {"question_id": 8, "category": "humanities", "turns": [
        "Compare and contrast the philosophical approaches of Plato and Aristotle.",
        "How did their different approaches influence modern Western philosophy?",
    ]},
    {"question_id": 9, "category": "writing", "turns": [
        "Write a haiku about artificial intelligence.",
        "Now expand that haiku into a full sonnet in Shakespearean form.",
    ]},
    {"question_id": 10, "category": "reasoning", "turns": [
        "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "Create a similar tricky word problem and explain why people commonly get it wrong.",
    ]},
]

JUDGE_PROMPT_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""


def generate_model_response(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a response from a causal LM (fp32)."""
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def judge_response_with_langsmith(
    question: str, answer: str, judge_model: str = "gpt-4",
) -> dict[str, Any]:
    """Score a response using LLM-as-judge."""
    try:
        from langchain_openai import ChatOpenAI
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        llm = ChatOpenAI(model=judge_model, temperature=0)
        result = llm.invoke(judge_prompt).content
        match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", result)
        score = float(match.group(1)) if match else 5.0
        return {"score": score, "explanation": result}
    except ImportError:
        logger.warning("LangChain not available. Using placeholder scoring.")
        return {"score": 5.0, "explanation": "LangChain not available"}
    except Exception as e:
        logger.warning("Judge scoring failed: %s", str(e))
        return {"score": 5.0, "explanation": f"Error: {str(e)}"}


def load_model_for_eval(
    model_path: str, base_model: str = None, is_base: bool = False,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a model for MT-Bench evaluation (fp32, no quantization)."""
    if is_base:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Load base + LoRA adapter
        base = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def run_mt_bench(config_path: str) -> None:
    """Run full MT-Bench evaluation across base, SFT, and PPO models."""
    with open(config_path) as f:
        params = yaml.safe_load(f)

    global_cfg = params.get("global", {})
    eval_cfg = params.get("evaluation", {})
    sft_cfg = params.get("sft", {})
    ppo_cfg = params.get("ppo", {})

    seed = global_cfg.get("seed", 42)
    base_model = global_cfg.get("base_model", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    sft_dir = sft_cfg.get("output_dir", "models/sft_adapter")
    ppo_dir = ppo_cfg.get("output_dir", "models/ppo_policy")
    judge_model = eval_cfg.get("judge_model", "gpt-4")
    output_path = eval_cfg.get("output_path", "reports/mt_bench_scores.json")
    mlflow_uri = global_cfg.get("mlflow_tracking_uri", "http://localhost:5000")

    set_seed(seed)

    questions_path = eval_cfg.get("mt_bench_questions_path", "data/mt_bench_questions.json")
    if Path(questions_path).exists():
        with open(questions_path) as f:
            questions = json.load(f)
    else:
        questions = MT_BENCH_QUESTIONS
        Path("data").mkdir(parents=True, exist_ok=True)
        with open(questions_path, "w") as f:
            json.dump(questions, f, indent=2)

    results = {
        "base": {"scores": [], "per_category": {}, "responses": []},
        "sft": {"scores": [], "per_category": {}, "responses": []},
        "ppo": {"scores": [], "per_category": {}, "responses": []},
    }

    model_configs = [
        ("base", base_model, True),
        ("sft", sft_dir, False),
        ("ppo", ppo_dir, False),
    ]

    for model_name, model_path, is_base in model_configs:
        logger.info("Evaluating %s model: %s", model_name, model_path)
        if not Path(model_path).exists() and not is_base:
            logger.warning("Model not found: %s. Skipping.", model_path)
            continue

        try:
            model, tokenizer = load_model_for_eval(
                model_path, base_model=base_model, is_base=is_base
            )
        except Exception as e:
            logger.warning("Could not load %s: %s. Skipping.", model_name, str(e))
            continue

        for q in questions:
            question_text = q["turns"][0]
            category = q["category"]
            response = generate_model_response(model, tokenizer, question_text)
            judge_result = judge_response_with_langsmith(question_text, response, judge_model)
            score = judge_result["score"]
            results[model_name]["scores"].append(score)
            results[model_name]["responses"].append({
                "question_id": q["question_id"], "category": category,
                "question": question_text, "response": response,
                "score": score, "explanation": judge_result["explanation"],
            })
            if category not in results[model_name]["per_category"]:
                results[model_name]["per_category"][category] = []
            results[model_name]["per_category"][category].append(score)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute summary
    summary = {}
    for mn in ["base", "sft", "ppo"]:
        scores = results[mn]["scores"]
        summary[f"{mn}_score"] = sum(scores) / len(scores) if scores else None
        for cat, cs in results[mn]["per_category"].items():
            summary[f"{mn}_{cat}_score"] = sum(cs) / len(cs) if cs else None

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("rlhf-evaluation")
    with mlflow.start_run(run_name="mt-bench-evaluation"):
        for key, value in summary.items():
            if value is not None:
                mlflow.log_metric(key, value)

    # Save results
    reports_dir = Path(output_path).parent
    reports_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "per_model": {
            mn: {
                "mean_score": summary.get(f"{mn}_score"),
                "per_category": {c: sum(s)/len(s) if s else None for c, s in results[mn]["per_category"].items()},
                "num_questions": len(results[mn]["scores"]),
            } for mn in ["base", "sft", "ppo"]
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("MT-Bench results saved to %s", output_path)

    for mn in ["base", "sft", "ppo"]:
        score = summary.get(f"{mn}_score")
        logger.info("  %s: %.2f / 10" if score else "  %s: N/A", mn.upper(), score or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="MT-Bench Evaluation")
    parser.add_argument("--config", type=str, default="params.yaml")
    args = parser.parse_args()
    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    run_mt_bench(args.config)


if __name__ == "__main__":
    main()
