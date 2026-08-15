import json
import time
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

# The 8B model produces false-negative faithfulness scores on long (~7.5k
# char) context — verified directly, it misses claims that are genuinely
# supported elsewhere in the context. The 70B model judges the same pairs
# correctly. Judging is a one-time eval cost, so the larger model is worth it.
JUDGE_MODEL = LLM_MODEL

judge_client = Groq(api_key=GROQ_API_KEY)


def load_eval_dataset(path="eval_dataset.json"):
    with open(path) as f:
        return json.load(f)


def citation_recall(actual_pmids, expected_pmids):
    if not expected_pmids:
        return None
    hits = set(actual_pmids) & set(expected_pmids)
    return len(hits) / len(expected_pmids)


def keyword_accuracy(actual_answer, expected_keywords):
    if not expected_keywords:
        return None
    answer_lower = actual_answer.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    return len(hits) / len(expected_keywords)


def is_refusal(answer: str) -> bool:
    markers = [
        "don't have enough reliable evidence",
        "don't have reliable information",
        "timed out",
        "couldn't find high-confidence",
    ]
    return any(m in answer.lower() for m in markers)


def _extract_score(judge_text: str) -> float:
    """Pull the first float in [0,1] out of the judge's response. Defaults to 0.0 if unparseable."""
    match = re.search(r"([01](?:\.\d+)?)", judge_text)
    return float(match.group(1)) if match else 0.0


def judge_faithfulness_and_relevancy(query: str, answer: str, context: str) -> tuple:
    """
    Custom LLM-as-judge (Groq llama-3.3-70b), not the `ragas` library — same
    concepts (faithfulness, answer relevancy), no OpenAI dependency. One
    combined call per query, not two, to stay inside the Groq free-tier
    daily token budget.
    """
    if is_refusal(answer):
        return None, None  # undefined for a refusal — nothing was claimed

    prompt = (
        "You are grading a clinical RAG answer on two dimensions.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "1. FAITHFULNESS — is every claim in the answer directly supported by the context?\n"
        "   1.0 = fully supported, 0.5 = partially supported, 0.0 = contradicts or absent from context.\n"
        "2. RELEVANCY — does the answer address the question asked?\n"
        "   1.0 = directly answers it, 0.5 = partially relevant, 0.0 = does not address it.\n\n"
        "Respond with EXACTLY two lines, nothing else:\n"
        "FAITHFULNESS: <0.0|0.5|1.0>\n"
        "RELEVANCY: <0.0|0.5|1.0>"
    )
    response = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = response.choices[0].message.content
    faith_match = re.search(r"FAITHFULNESS:\s*([01](?:\.\d+)?)", text)
    relev_match = re.search(r"RELEVANCY:\s*([01](?:\.\d+)?)", text)
    faithfulness = float(faith_match.group(1)) if faith_match else _extract_score(text)
    relevancy = float(relev_match.group(1)) if relev_match else None
    return faithfulness, relevancy


def judge_answer(result, item, context: str):
    scores = {}

    scores["keyword_accuracy"] = keyword_accuracy(result["answer"], item["expected_keywords"])
    scores["citation_recall"] = citation_recall(result["sources"], item["expected_pmids"])
    scores["answered"] = not is_refusal(result["answer"])

    # Refusal-correctness: for difficulty="refusal" items, the RIGHT answer is to refuse.
    if item["difficulty"] == "refusal":
        scores["refusal_correct"] = is_refusal(result["answer"])
    else:
        scores["refusal_correct"] = None

    faithfulness, relevancy = judge_faithfulness_and_relevancy(item["query"], result["answer"], context)
    scores["faithfulness"] = faithfulness
    scores["relevancy"] = relevancy

    return scores


def run_eval(dataset, agent_run_fn, checkpoint_path="eval_checkpoint.json"):
    """
    Resumable: writes progress to checkpoint_path after every query, so a
    mid-run failure (e.g. hitting the Groq free-tier daily token cap) loses
    at most one query's work, not the whole run. Re-running with the same
    checkpoint_path picks up where it left off.
    """
    try:
        with open(checkpoint_path) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    done_queries = {r["query"] for r in results}

    for i, item in enumerate(dataset):
        query = item["query"]
        if query in done_queries:
            continue

        print(f"\n[{i+1}/{len(dataset)}] {query[:60]}")

        start = time.time()
        output = agent_run_fn(query)
        latency = round(time.time() - start, 2)

        scores = judge_answer(output, item, context=output.get("context_text", ""))

        results.append({
            "query": query,
            "query_type": item["query_type"],
            "difficulty": item["difficulty"],
            "answer": output["answer"][:200] if output["answer"] else "",
            "sources": output["sources"],
            "crag_score": output["crag_score"],
            "attempts": output["attempts"],
            "latency_s": latency,
            **scores,
        })

        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2)

        time.sleep(0.3)  # be polite to the Groq API

    return results


def _avg(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def generate_report(results, out_path="eval_report.json"):
    print("\n" + "=" * 90)
    header = f"{'#':<3} {'Type':<10} {'Diff':<9} {'KW%':<6} {'Cite%':<7} {'Faith':<7} {'Relev':<7} {'Ans?':<5} {'Lat'}"
    print(header)
    print("-" * 90)

    for i, r in enumerate(results, 1):
        kw = f"{r['keyword_accuracy']:.0%}" if r["keyword_accuracy"] is not None else "N/A"
        cite = f"{r['citation_recall']:.0%}" if r["citation_recall"] is not None else "N/A"
        faith = f"{r['faithfulness']:.1f}" if r["faithfulness"] is not None else "N/A"
        relev = f"{r['relevancy']:.1f}" if r["relevancy"] is not None else "N/A"
        ans = "YES" if r["answered"] else "NO"
        print(f"{i:<3} {r['query_type']:<10} {r['difficulty']:<9} {kw:<6} {cite:<7} {faith:<7} {relev:<7} {ans:<5} {r['latency_s']}s")

    summary = {
        "n_queries": len(results),
        "avg_keyword_accuracy": _avg([r["keyword_accuracy"] for r in results]),
        "avg_citation_recall": _avg([r["citation_recall"] for r in results]),
        "avg_faithfulness": _avg([r["faithfulness"] for r in results]),
        "avg_relevancy": _avg([r["relevancy"] for r in results]),
        "answer_rate": sum(r["answered"] for r in results) / len(results),
        "refusal_accuracy": _avg([r["refusal_correct"] for r in results]),
        "avg_latency_s": _avg([r["latency_s"] for r in results]),
        "p95_latency_s": sorted(r["latency_s"] for r in results)[int(len(results) * 0.95) - 1] if len(results) > 1 else results[0]["latency_s"],
    }

    print("=" * 90)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:<22}: {v:.3f}")
        else:
            print(f"{k:<22}: {v}")

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSaved full report to {out_path}")

    return summary
