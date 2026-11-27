import os
import sys
import csv
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

# ---------------- CONFIG ----------------
MODEL = "gemini-2.5-flash"
CSV_FILE = "gsm8k_train.csv"        # your English data
OUTPUT_FILE = "train_bn.csv"        # translated output
QUESTION_COL = "question"           # column names in your input CSV
ANSWER_COL = "answer"
KEYS_FILE = "api_keys.txt"          # or set env var GEMINI_API_KEYS=key1,key2,...
WORKERS = 4
RATE_LIMIT_COOLDOWN = 60
MAX_OTHER_RETRIES = 6

OUTPUT_COLUMNS = ["idx", QUESTION_COL, ANSWER_COL, "question_bn", "answer_bn"]

PROMPT = """You are a professional translator. Translate the following math word problem and its step-by-step solution from English into natural, fluent Bangla (Bengali script).

Rules:
- Write ALL numbers using Bangla digits (০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯). For example, 48 becomes ৪৮ and 2.5 becomes ২.৫.
- Keep the same numeric values exactly. Do not change, round, or recalculate anything.
- Keep math expressions, operators, and calculator annotations inside << >> with the same structure (only the digits change to Bangla digits).
- Keep the final line "#### <number>" with the same number, written in Bangla digits.
- Keep line breaks in the answer the same as the original.
- Translate only the natural-language text.

Return ONLY a JSON object with keys "question" and "answer", with the Bangla translations as string values.

Input:
"""

EN_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def to_bn_digits(text):
    return text.translate(EN_TO_BN)


def to_en_digits(text):
    return text.translate(BN_TO_EN)


def final_number(text):
    m = re.search(r"####\s*(\S+)", text)
    return m.group(1) if m else None


# ---------------- KEY POOL ----------------
class KeyPool:
    """Round-robin over API keys; keys that hit rate limits rest for a while."""

    def __init__(self, keys):
        self.keys = keys
        self.clients = {k: genai.Client(api_key=k) for k in keys}
        self.cooldown_until = {k: 0.0 for k in keys}
        self.lock = threading.Lock()
        self.idx = 0

    def get(self):
        while True:
            with self.lock:
                now = time.time()
                n = len(self.keys)
                for _ in range(n):
                    key = self.keys[self.idx % n]
                    self.idx += 1
                    if self.cooldown_until[key] <= now:
                        return key
                wait = min(self.cooldown_until.values()) - now
            time.sleep(max(wait, 1))

    def penalize(self, key, seconds):
        with self.lock:
            self.cooldown_until[key] = time.time() + seconds


def load_keys():
    env = os.getenv("GEMINI_API_KEYS")
    if env:
        keys = [k.strip() for k in env.split(",") if k.strip()]
    elif os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            keys = [line.strip() for line in f if line.strip()]
    else:
        raise SystemExit(f"No API keys found. Create {KEYS_FILE} or set GEMINI_API_KEYS.")
    if not keys:
        raise SystemExit("Key list is empty.")
    print(f"Loaded {len(keys)} API key(s).")
    return list(dict.fromkeys(keys))


# ---------------- TRANSLATION ----------------
def translate_item(pool, item):
    import json
    payload = json.dumps(
        {"question": item[QUESTION_COL], "answer": item[ANSWER_COL]},
        ensure_ascii=False,
    )
    prompt = PROMPT + payload
    other_errors = 0

    while True:
        key = pool.get()
        try:
            resp = pool.clients[key].models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            data = json.loads(resp.text)
            q_bn = to_bn_digits(data["question"].strip())
            a_bn = to_bn_digits(data["answer"].strip())

            if not q_bn or "####" not in a_bn:
                raise ValueError("Malformed translation")

            if to_en_digits(final_number(a_bn) or "") != final_number(item[ANSWER_COL]):
                raise ValueError("Final answer mismatch")

            return q_bn, a_bn

        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "429" in msg or "resource_exhausted" in low or "quota" in low:
                pool.penalize(key, RATE_LIMIT_COOLDOWN)
            elif "api_key_invalid" in low or "permission_denied" in low or "403" in msg:
                print(f"[!] Key ...{key[-6:]} looks invalid, disabling it.")
                pool.penalize(key, 10**9)
            else:
                other_errors += 1
                if other_errors > MAX_OTHER_RETRIES:
                    print(f"[x] Giving up on item after retries: {msg[:120]}")
                    return None, None
                time.sleep(min(2 ** other_errors, 30) + random.random())


# ---------------- DATA & RESUME ----------------
def load_csv_rows():
    if not os.path.exists(CSV_FILE):
        raise SystemExit(f"CSV not found: {CSV_FILE}")
    with open(CSV_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {QUESTION_COL, ANSWER_COL} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing column(s) in CSV: {missing}. Found: {reader.fieldnames}")
        return list(reader)


def load_done_ids():
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done.add(int(row["idx"]))
                except (KeyError, TypeError, ValueError):
                    pass
    return done


def ensure_output_header():
    if not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(OUTPUT_COLUMNS)


# ---------------- MAIN ----------------
def main():
    keys = load_keys()
    pool = KeyPool(keys)
    data = load_csv_rows()
    ensure_output_header()
    done = load_done_ids()
    todo = [(i, row) for i, row in enumerate(data) if i not in done]
    print(f"Total: {len(data)} | Already done: {len(done)} | Remaining: {len(todo)}")

    write_lock = threading.Lock()
    failed = []

    def work(idx, row):
        q_bn, a_bn = translate_item(pool, row)
        if q_bn is None:
            return idx, False
        with write_lock:
            with open(OUTPUT_FILE, "a", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerow([
                    idx,
                    row[QUESTION_COL],
                    row[ANSWER_COL],
                    q_bn,
                    a_bn,
                ])
                f.flush()
        return idx, True

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(work, i, row) for i, row in todo]
        completed = 0
        for fut in as_completed(futures):
            idx, ok = fut.result()
            completed += 1
            if not ok:
                failed.append(idx)
            if completed % 50 == 0 or completed == len(futures):
                print(f"Progress: {completed}/{len(futures)}")

    print(f"\nFinished. Output: {OUTPUT_FILE}")
    if failed:
        print(f"Failed items (re-run script to retry them): {failed}")


if __name__ == "__main__":
    main()