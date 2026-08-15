from evals import load_eval_dataset, run_eval, generate_report
from agent import run_agent as run

if __name__ == "__main__":
    print("Loading eval dataset...")
    dataset = load_eval_dataset()

    print(f"Running {len(dataset)} questions through the CRAG agent...")
    results = run_eval(dataset, agent_run_fn=run)

    print("\nGenerating report...")
    generate_report(results)
