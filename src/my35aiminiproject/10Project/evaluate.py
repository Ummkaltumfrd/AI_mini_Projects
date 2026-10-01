import json
import subprocess
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "dataset" / "test.jsonl"
RESULTS_FILE = BASE_DIR / "evaluation_results.json"


# --------------------------------------------------
# Models
# --------------------------------------------------

FINE_TUNED_MODEL = "my_model"
BASE_MODEL = "llama3.2:1b"


# --------------------------------------------------
# Load test examples
# --------------------------------------------------

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test dataset not found:\n{TEST_FILE}")

test_examples = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        test_examples.append(json.loads(line))


# Use only 5 held-out examples
test_examples = test_examples[:5]

print(f"Number of test examples: {len(test_examples)}")


# --------------------------------------------------
# Run a model with Ollama
# --------------------------------------------------

def run_model(model_name, prompt):
    result = subprocess.run(
        ["ollama", "run", model_name, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}"

    return result.stdout.strip()


# --------------------------------------------------
# Evaluate both models
# --------------------------------------------------

results = []

for i, example in enumerate(test_examples, start=1):

    instruction = str(example["instruction"]).strip()
    expected_response = str(example["response"]).strip()

    print("\n" + "=" * 70)
    print(f"TEST EXAMPLE {i}")
    print("=" * 70)

    print("\nPrompt:")
    print(instruction)

    print("\nRunning fine-tuned model...")

    fine_tuned_response = run_model(
        FINE_TUNED_MODEL,
        instruction,
    )

    print("\nFine-tuned response:")
    print(fine_tuned_response)

    print("\nRunning base model...")

    base_response = run_model(
        BASE_MODEL,
        instruction,
    )

    print("\nBase model response:")
    print(base_response)

    results.append(
        {
            "test_number": i,
            "instruction": instruction,
            "expected_response": expected_response,
            "fine_tuned_response": fine_tuned_response,
            "base_response": base_response,
        }
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULTS_FILE)