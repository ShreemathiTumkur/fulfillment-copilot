import csv, subprocess, re, sys, pathlib, textwrap

# ------------------------------------------------------------------
# Canonical keyword → list of acceptable variants
# ------------------------------------------------------------------
SYN = {
    "weather":  ["weather", "rain", "storm", "snow"],
    "traffic":  ["traffic", "congestion", "road"],
    "inventory": ["inventory", "stock", "out-of-stock"],
    "unknown":  ["unknown"]
}


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
QA_FILE = pathlib.Path("evals/qa_seed.csv")        # question,expected-keyword
SCRIPT  = pathlib.Path("scripts/rag_search.py")    # your RAG answerer
TOP_K   = "3"                                     # how many passages to pass

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def call_rag(question: str) -> str:
    """
    Run rag_search.py and return only the LLM answer section.
    """
    result = subprocess.run(
        ["python", SCRIPT, question, "-k", TOP_K],
        capture_output=True, text=True, check=True
    )
    # Capture everything after the line that starts 'Answer:'
    match = re.search(r"^Answer:\s*\n([\s\S]*)", result.stdout, re.M)
    return match.group(1).strip() if match else ""


def evaluate():
    total, hits = 0, 0

    with QA_FILE.open() as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip blank or malformed lines
            if len(row) < 2:
                continue

            question, keyword = row[0].strip(), row[1].strip()
            total += 1

            answer = call_rag(question)
            first = next((ln.strip().lower() for ln in answer.splitlines() if ln.strip()), "")
            first = first.replace("reason:", "").strip()       # in case "Reason:" sneaks back in

            detected = None
            for key, variants in SYN.items():
                if any(first.startswith(v) for v in variants):
                   detected = key
                   break

            ok = detected == keyword.lower()

            hits += ok

            print(f"Q: {question}")
            print(f"Expected keyword: {keyword}")
            print("Answer:", textwrap.shorten(answer, 140))
            print("✔️  PASS\n" if ok else "❌  FAIL\n")

    accuracy = hits / total if total else 0
    print(f"Accuracy: {hits}/{total} = {accuracy:.0%}")

# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if not QA_FILE.exists():
        sys.exit(f"❌  Eval file not found: {QA_FILE}")
    evaluate()    
