"""
Main experiment runner for HyPo-RAG.

Runs the complete experiment protocol using RoG datasets from HuggingFace:
1. Load dataset with per-question subgraphs
2. Validation hyperparameter tuning (optional)
3. Main evaluation on test set
4. Ablation studies (optional)

No global KG-Trie needed - each question has its own subgraph!
"""

import argparse
from pathlib import Path
import json
from datetime import datetime
import sys
from tqdm import tqdm
from datasets import load_dataset

from src.config import Config
from src.inference import HyPoRAG
from src.eval import compute_metrics
from src.utils import logger

from src.config import Config
from src.inference import HyPoRAG
from src.eval import compute_metrics
from src.utils import logger, save_results


def run_evaluation(config: Config, split: str = "test", max_examples: int = None):
    """
    Run HyPo-RAG evaluation on dataset split.
    
    Args:
        config: Configuration object
        split: Dataset split ("train", "validation", "test")
        max_examples: Maximum number of examples to evaluate (None = all)
        
    Returns:
        Dictionary with results
    """
    logger.info("="*80)
    logger.info(f"Running HyPo-RAG evaluation on {config.dataset_name} ({split})")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}...")
    ds = load_dataset(config.dataset_name)
    data = ds[split]
    
    if max_examples:
        data = data.select(range(min(max_examples, len(data))))
    
    logger.info(f"Evaluating {len(data)} examples...")
    
    # Initialize HyPo-RAG
    hyporag = HyPoRAG(config)
    
    # Run evaluation
    results = []
    aggregate_metrics = {
        "hit@1": 0.0,
        "f1": 0.0,
        "em": 0.0,
        "total_time": 0.0,
        "avg_time_per_q": 0.0,
        "total_api_calls": 0,
        "avg_api_calls": 0.0,
        "total_tokens": 0,
        "avg_tokens": 0.0,
    }
    
    for example in tqdm(data, desc="Evaluating"):
        question = example["question"]
        gold_answers = example["answer"]
        graph_triples = example["graph"]
        question_entities = example["q_entity"]
        
        try:
            # Run HyPo-RAG
            result = hyporag.run(question, graph_triples, question_entities)
            
            # Compute metrics
            metrics = compute_metrics(result, gold_answers)
            
            # Aggregate metrics
            aggregate_metrics["hit@1"] += metrics["hit@1"]
            aggregate_metrics["f1"] += metrics["f1"]
            aggregate_metrics["em"] += metrics["em"]
            aggregate_metrics["total_time"] += result.metadata["timings"]["total"]
            aggregate_metrics["total_api_calls"] += result.metadata["api_calls"]
            aggregate_metrics["total_tokens"] += result.metadata["tokens"]["total"]
            
            # Save result
            results.append({
                "question_id": example["id"],
                "question": question,
                "gold_answers": gold_answers,
                "predicted_answer": result.answer,
                "metrics": metrics,
                "metadata": result.metadata,
            })
            
        except Exception as e:
            logger.error(f"Error on question {example['id']}: {e}")
            results.append({
                "question_id": example["id"],
                "question": question,
                "error": str(e),
            })
    
    # Compute averages
    n = len([r for r in results if "error" not in r])
    if n > 0:
        aggregate_metrics["hit@1"] /= n
        aggregate_metrics["f1"] /= n
        aggregate_metrics["em"] /= n
        aggregate_metrics["avg_time_per_q"] = aggregate_metrics["total_time"] / n
        aggregate_metrics["avg_api_calls"] = aggregate_metrics["total_api_calls"] / n
        aggregate_metrics["avg_tokens"] = aggregate_metrics["total_tokens"] / n
    
    logger.info(f"\nResults:")
    logger.info(f"  Hit@1: {aggregate_metrics['hit@1']:.3f}")
    logger.info(f"  F1: {aggregate_metrics['f1']:.3f}")
    logger.info(f"  EM: {aggregate_metrics['em']:.3f}")
    logger.info(f"  Avg time/question: {aggregate_metrics['avg_time_per_q']:.2f}s")
    logger.info(f"  Avg API calls/question: {aggregate_metrics['avg_api_calls']:.1f}")
    logger.info(f"  Avg tokens/question: {aggregate_metrics['avg_tokens']:.0f}")
    
    return {
        "config": config.__dict__,
        "dataset": config.dataset_name,
        "split": split,
        "num_examples": len(results),
        "aggregate_metrics": aggregate_metrics,
        "per_question_results": results,
    }


def validation_phase(config: Config, validation_size: int = 100):
    """
    Optional: Validation hyperparameter tuning.
    
    Grid search over (α, β, γ, λ) on validation questions.
    """
    logger.info("="*80)
    logger.info("Validation Hyperparameter Tuning")
    logger.info("="*80)
    
    # Define grid
    alpha_range = [0.3, 0.4, 0.5]
    beta_range = [0.2, 0.3, 0.4]
    gamma_range = [0.1, 0.2, 0.3]
    lambda_range = [0.001, 0.002, 0.003]
    
    best_config = config
    best_f1 = 0.0
    
    total_configs = len(alpha_range) * len(beta_range) * len(gamma_range) * len(lambda_range)
    logger.info(f"Testing {total_configs} configurations on {validation_size} validation examples...")
    
    for alpha in alpha_range:
        for beta in beta_range:
            for gamma in gamma_range:
                for lambda_ in lambda_range:
                    test_config = Config()
                    test_config.alpha = alpha
                    test_config.beta = beta
                    test_config.gamma = gamma
                    test_config.lambda_ = lambda_
                    
                    # Evaluate
                    results = run_evaluation(test_config, split="validation", max_examples=validation_size)
                    f1 = results["aggregate_metrics"]["f1"]
                    
                    logger.info(f"α={alpha}, β={beta}, γ={gamma}, λ={lambda_}: F1={f1:.3f}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = test_config
    
    logger.info(f"\nBest config: α={best_config.alpha}, β={best_config.beta}, "
               f"γ={best_config.gamma}, λ={best_config.lambda_} (F1={best_f1:.3f})")
    
    return best_config


def ablation_studies(base_config: Config, ablation_size: int = 200):
    """
    Optional: Ablation studies.
    
    Test variants: w/o HGP, w/o budget, w/o diversity, w/o veracity.
    """
    logger.info("="*80)
    logger.info("Ablation Studies")
    logger.info("="*80)
    
    ablations = {
        "Full Model": {},
        "w/o HGP": {"num_hypotheses": 0, "num_subquestions": 0},
        "w/o Budget": {"token_budget": 10000},
        "w/o Diversity": {"beta": 0.0},
        "w/o Veracity": {"gamma": 0.0},
    }
    
    ablation_results = {}
    
    for name, modifications in ablations.items():
        logger.info(f"\nRunning: {name}")
        
        # Create config with modifications
        ablation_config = Config()
        # Copy base config attributes
        for attr in dir(base_config):
            if not attr.startswith("_"):
                try:
                    setattr(ablation_config, attr, getattr(base_config, attr))
                except:
                    pass
        
        # Apply modifications
        for key, value in modifications.items():
            setattr(ablation_config, key, value)
        
        # Run evaluation
        results = run_evaluation(ablation_config, split="test", max_examples=ablation_size)
        ablation_results[name] = results["aggregate_metrics"]
        
        logger.info(f"  Hit@1: {results['aggregate_metrics']['hit@1']:.3f}")
        logger.info(f"  F1: {results['aggregate_metrics']['f1']:.3f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Hit@1':<10} {'F1':<10} {'Time(s)':<12} {'Tokens':<10}")
    print("-"*80)
    
    for name, metrics in ablation_results.items():
        print(f"{name:<20} {metrics['hit@1']:<10.3f} {metrics['f1']:<10.3f} "
              f"{metrics['avg_time_per_q']:<12.2f} {metrics['avg_tokens']:<10.0f}")
    
    print("="*80)
    
    return ablation_results


def print_main_results_table(metrics: dict):
    """Print main results in table format (Table 1)."""
    print("\n" + "="*80)
    print("TABLE 1: Main Results on WebQSP Test Set")
    print("="*80)
    print(f"{'Method':<30} {'Hit@1':<10} {'F1':<10} {'Tokens/Q':<12} {'Calls/Q':<10} {'Time(s)':<10}")
    print("-"*80)
    
    # HyPo-RAG (our method)
    print(f"{'HyPo-RAG (ours)':<30} {metrics['hit@1']:.3f}      {metrics['f1']:.3f}      "
          f"{metrics['avg_tokens']:<12.1f} {metrics['avg_api_calls']:<10.1f} {metrics['avg_time_per_q']:<10.2f}")
    
    # Baselines (reported from papers)
    baselines = [
        ("GCR (fine-tuned)", 0.926, None, 231, 2.0, 3.6),
        ("GNN-RAG+RA (trained)", 0.907, None, None, None, None),
        ("SubgraphRAG (train-free)", 0.889, None, None, None, None),
        ("iQUEST (train-free)", 0.889, None, None, None, None),
        ("GIVE (train-free)", 0.880, None, None, None, None),
        ("ToG (train-free)", 0.850, None, 7069, 11.6, 16.14),
        ("RoG (train-free)", 0.870, None, 521, 2.0, 2.6),
    ]
    
    for name, hit1, f1, tokens, calls, time in baselines:
        hit1_str = f"{hit1:.3f}" if hit1 else "-"
        f1_str = f"{f1:.3f}" if f1 else "-"
        tokens_str = f"{tokens:.1f}" if tokens else "-"
        calls_str = f"{calls:.1f}" if calls else "-"
        time_str = f"{time:.2f}" if time else "-"
        
        print(f"{name:<30} {hit1_str:<10} {f1_str:<10} {tokens_str:<12} {calls_str:<10} {time_str:<10}")
    
    print("="*80)
    print("Note: Baseline results reported from original papers")
    print("="*80 + "\n")


def print_ablation_table(results: dict):
    """Print ablation results in table format (Table 2)."""
    print("\n" + "="*60)
    print("TABLE 2: Ablation Study")
    print("="*60)
    print(f"{'Variant':<30} {'Hit@1':<15} {'Δ Hit@1':<15}")
    print("-"*60)
    
    # Assume first result is full model
    baseline = list(results.values())[0]["hit@1"]
    
    for name, metrics in results.items():
        hit1 = metrics["hit@1"]
        delta = hit1 - baseline
        print(f"{name:<30} {hit1:.3f}           {delta:+.3f}")
    
    print("="*60 + "\n")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run HyPo-RAG experiments")
    parser.add_argument("--dataset", type=str, default="rmanluo/RoG-webqsp",
                       help="HuggingFace dataset name (default: rmanluo/RoG-webqsp)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                       help="Dataset split to evaluate (default: test)")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to evaluate (default: all)")
    parser.add_argument("--validation", action="store_true",
                       help="Run validation hyperparameter tuning first")
    parser.add_argument("--ablations", action="store_true",
                       help="Run ablation studies")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for results (default: ./results)")
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.dataset_name = args.dataset
    config.results_dir = args.output_dir
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.results_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("HyPo-RAG EXPERIMENTS")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Max examples: {args.max_examples or 'all'}")
    logger.info(f"Results directory: {exp_dir}")
    logger.info("="*80)
    
    # Phase 1: Optional validation tuning
    if args.validation:
        best_config = validation_phase(config, validation_size=100)
        config = best_config
        
        # Save tuned config
        with open(exp_dir / "tuned_config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
    
    # Phase 2: Main evaluation
    results = run_evaluation(config, split=args.split, max_examples=args.max_examples)
    
    # Save results
    results_file = exp_dir / f"{args.split}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_file}")
    
    # Print table
    print_main_results_table(results["aggregate_metrics"])
    
    # Phase 3: Optional ablations
    if args.ablations:
        ablation_results = ablation_studies(config, ablation_size=200)
        
        # Save ablation results
        ablation_file = exp_dir / "ablation_results.json"
        with open(ablation_file, "w") as f:
            json.dump(ablation_results, f, indent=2)
        logger.info(f"\nAblation results saved to: {ablation_file}")
    
    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETE!")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
