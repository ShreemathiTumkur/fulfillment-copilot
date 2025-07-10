import csv, subprocess, re, sys, pathlib, textwrap

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
    Run rag_search.py as a subprocess and capture the LLM's final answer.
    The answer is everything AFTER the line that starts with 'Answer:'.
    """
    result = subprocess.run(
        ["python", SCRIPT, question, "-k", TOP_K],
        capture_output=True, text=True, check=True
    )
    # Regex: grab text after the --- line and 'Answer:'
    match = re.search(r"^---\s*Answer:\s*\n(.*)", result.stdout, re.S)
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
            ok = re.search(keyword, answer, re.I) is not None
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
