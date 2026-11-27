"""
Translate GSM8K train (English) -> Bangla using TranslateGemma 4b via Ollama.

Install:  pip install requests tqdm
Run:      python translate_gsm8k_bn.py

Output:   train_bn.csv  (columns: question_bn, answer_bn)
Resumable: if you stop the script, run it again and it continues where it left off.
"""

import csv
import os
import re

import requests
from tqdm import tqdm

# =============================================================================
#  SETTINGS  <-- change these
# =============================================================================
INPUT_CSV = "C:\\Users\\EKRAJ\\OneDrive\\Documents\\GitHub\\Do-Reasoning-Model-Knows-When-To-Stop-\\data\\gsm8k train.csv"   # <-- PUT YOUR ENGLISH train.csv PATH HERE
OUTPUT_CSV = "train_bn.csv"                 # translated file (saved next to this script)
ISSUES_LOG = "translation_issues.txt"       # row numbers whose numbers didn't validate

QUESTION_COLUMN = "question"                # column names in your input CSV
ANSWER_COLUMN = "answer"

MODEL = "translategemma:4b"                 # check the exact name with: ollama list
OLLAMA_URL = "http://localhost:11434/api/chat"

LIMIT = None                                # None = all rows, or e.g. 20 for a quick test
MAX_RETRIES = 3                             # retries if a translation fails validation
# =============================================================================


PROMPT = """You are a professional translator. Translate the following math word problem and its step-by-step solution from English into natural, fluent Bangla (Bengali script).

Rules:
- Write ALL numbers using Bangla digits (০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯). For example, 48 becomes ৪৮ and 2.5 becomes ২.৫.
- Keep the same numeric values exactly. Do not change, round, or recalculate anything.
- Keep math expressions, operators, and calculator annotations inside << >> with the same structure (only the digits change to Bangla digits).
- Keep the final line "#### <number>" with the same number, written in Bangla digits.
- Keep line breaks in the answer the same as the original.
- Translate only the natural-language text."""

# Tells the model how to format its reply so we can split question and answer.
FORMAT_INSTRUCTION = """
Output format (exactly, no extra commentary):
### QUESTION
<translated question>
### ANSWER
<translated solution>
"""


# ---------- Step 1: ask the model to translate one example -------------------
def translate_once(question, answer):
    message = f"{PROMPT}\n{FORMAT_INSTRUCTION}\n### QUESTION\n{question}\n### ANSWER\n{answer}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1536},
        },
        timeout=300,
    )
    response.raise_for_status()
    text = response.json()["message"]["content"].strip()

    # split the reply into question and answer
    match = re.search(r"###\s*QUESTION\s*\n(.*?)\n\s*###\s*ANSWER\s*\n(.*)", text, re.S)
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


# ---------- Step 2: check the numbers were preserved -------------------------
BANGLA_TO_ENGLISH = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def get_numbers(text):
    text = text.translate(BANGLA_TO_ENGLISH).replace(",", "")
    return re.findall(r"\d+(?:\.\d+)?", text)


def is_valid(q_en, a_en, q_bn, a_bn):
    # final "#### number" line must exist and match
    final_en = re.search(r"####\s*(.+)\s*$", a_en)
    final_bn = re.search(r"####\s*(.+)\s*$", a_bn)
    if not final_en or not final_bn:
        return False
    if get_numbers(final_en.group(1)) != get_numbers(final_bn.group(1)):
        return False
    if re.search(r"[0-9]", final_bn.group(1)):      # must use Bangla digits
        return False
    # all numbers must be the same, in the same order
    if get_numbers(q_en) != get_numbers(q_bn):
        return False
    if get_numbers(a_en) != get_numbers(a_bn):
        return False
    # calculator annotations << >> must be kept
    if a_en.count("<<") != a_bn.count("<<") or a_en.count(">>") != a_bn.count(">>"):
        return False
    return True


# ---------- Step 3: translate with retries -----------------------------------
def translate(question, answer):
    """Returns (question_bn, answer_bn, passed_validation)."""
    best = ("", "", False)
    for _ in range(MAX_RETRIES):
        try:
            result = translate_once(question, answer)
        except requests.RequestException as error:
            print(f"\n[warning] Ollama request failed: {error}")
            continue
        if result is None:
            continue
        q_bn, a_bn = result
        if is_valid(question, answer, q_bn, a_bn):
            return q_bn, a_bn, True
        if not best[0]:
            best = (q_bn, a_bn, False)      # keep first attempt as fallback
    return best


# ---------- Step 4: main loop ------------------------------------------------
def count_rows_done():
    """How many rows are already saved in the output CSV (for resuming)."""
    if not os.path.exists(OUTPUT_CSV):
        return 0
    with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)   # minus header row


def main():
    # read the English dataset
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if LIMIT:
        rows = rows[:LIMIT]

    start = count_rows_done()
    print(f"{len(rows)} rows total, {start} already translated.")

    mode = "a" if start > 0 else "w"
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8-sig") as out, \
         open(ISSUES_LOG, "a", encoding="utf-8") as log:

        writer = csv.writer(out)
        if start == 0:
            writer.writerow(["question_bn", "answer_bn"])

        for i in tqdm(range(start, len(rows)), initial=start, total=len(rows)):
            question = rows[i][QUESTION_COLUMN]
            answer = rows[i][ANSWER_COLUMN]

            q_bn, a_bn, ok = translate(question, answer)

            writer.writerow([q_bn, a_bn])
            out.flush()                      # saved right away, safe to Ctrl+C
            if not ok:
                log.write(f"{i}\n")
                log.flush()

    print(f"Done! Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()