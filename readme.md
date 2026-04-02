*This project has been created as part of the 42 curriculum by vborysov*


# Description

This project implements a function calling system powered by a small Large Language Model (LLM).
The goal is to translate natural language prompts into structured function calls instead of generating direct answers.

Instead of answering questions directly, the system determines:

- which function should be called
- which arguments should be passed
- how to format them according to a strict schema

Example:
Input
```
"What is the sum of 40 and 2?"
```

Output
```
{
  "function": "fn_add_numbers",
  "arguments": {
    "a": 40,
    "b": 2
  }
}
```

The system guarantees valid JSON output using constrained decoding techniques, ensuring reliability even with small models. The main challenge of this project is achieving reliable structured output using a very small model `(Qwen3-0.6B)` through constrained decoding.
## Algorithm Explanation — Constrained Decoding

The project implements a constrained decoding strategy on top of a small
causal language model using direct access to token logits.

Instead of letting the model freely generate text, token selection is
restricted programmatically to ensure that the output always follows a
valid JSON structure describing a function call.

### Model Interface

The system uses a lightweight wrapper (`Small_LLM_Model`) built on top of
Hugging Face Transformers. The wrapper provides:

- token encoding and decoding
- access to next-token logits
- automatic device selection (CPU / CUDA / MPS)
- inference-only execution (no gradients)

At each step, the generator queries:
```
get_logits_from_input_ids(...)
```

which returns raw logits for the next token.

---

### Structured Generation Strategy

JSON is not generated freely. Instead, the structure is assembled step by step:

1. A fixed JSON prefix is injected manually.
2. The model generates only variable parts:
   - function name
   - parameter values.
3. Token choices are restricted depending on the current generation phase.

The generator explicitly controls:

- when keys appear,
- when quotes open/close,
- when commas are inserted,
- when generation must stop.

---

### Constrained Token Sampling

When generating fields with a limited set of valid values (e.g. function names),
the algorithm performs masked decoding:

1. Logits for all tokens are retrieved.
2. Only tokens corresponding to allowed continuations are kept.
3. All other logits are replaced with a large negative value.
4. Greedy selection (`argmax`) chooses the next token.

This ensures that generated text can only match one of the allowed options.

The process effectively performs **prefix-constrained decoding**.

---

### Value Generation

Parameter values are generated depending on expected type:

- **string values**
  - opening quote is injected
  - tokens are generated until a closing quote appears
- **numeric values**
  - greedy decoding continues until a structural delimiter
    (`,`, `}`, whitespace) is reached.

---

### Determinism

Sampling is fully deterministic:

- no temperature
- no randomness
- greedy decoding only

This guarantees reproducible outputs.

---

### Result

The approach guarantees:

- syntactically valid JSON
- schema-aligned structure
- controlled generation using a small LLM

## Design Decisions

Several implementation choices were made to balance reliability,
simplicity, and transparency of the generation process.

### Direct access to model logits

Instead of using high-level text generation APIs from Hugging Face,
the implementation directly retrieves next-token logits from the model.
This allows explicit control over token selection and enables
constrained decoding.

This decision was essential because JSON validity cannot be guaranteed
with unconstrained generation.

---

### Hybrid generation approach

The JSON structure itself is not generated entirely by the model.
Instead:

- structural elements (braces, keys, commas) are injected manually,
- the LLM generates only semantic components:
  - function name
  - parameter values.

This separates responsibilities between deterministic code and
probabilistic model reasoning.

---

### Prefix-constrained decoding

When generating function names, only tokens that can extend one of the
valid candidates are allowed. This prevents invalid tool selection and
reduces hallucinations.

---

### Greedy decoding

Greedy decoding (`argmax`) was selected instead of sampling methods
(temperature, top-k, etc.) to ensure deterministic and reproducible
outputs.

Reliability was prioritized over linguistic diversity.

---

### Lightweight local inference

The project uses a small causal language model executed locally via a
custom SDK wrapper. This avoids external APIs and keeps resource usage
low while remaining compatible with different hardware setups.

## Performance Analysis

### Accuracy

The model performs well when prompts clearly describe an available
function.

Observed behavior:

- correct function selection in most cases
- reliable parameter extraction for simple inputs
- reduced accuracy for ambiguous or underspecified prompts

Because structure is enforced programmatically, semantic mistakes do not
break output validity.

---

### Reliability

Reliability is the main strength of the system.

The constrained decoding strategy guarantees:

- valid JSON syntax
- correct object structure
- presence of required fields

Even when generation fails, the system safely returns an empty JSON
object instead of producing malformed output.

---

### Speed

Generation speed depends on token-by-token inference:

- one model forward pass per generated token
- greedy decoding without sampling overhead

Execution time scales linearly with output length and number of prompts.
The provided dataset completes within a few minutes on standard hardware.

---

### Resource Usage

Using a small (~0.6B parameter) model allows execution on:

- CPU-only systems
- Apple Silicon (MPS)
- CUDA-enabled GPUs

Memory usage remains low compared to larger LLM solutions.

## Challenges Faced

### Enforcing JSON correctness

Language models naturally generate free-form text, which often results
in invalid JSON.

Solution:
generation was redesigned to construct JSON structure explicitly while
restricting token choices during variable fields.

---

### Tokenization mismatch

JSON symbols and function names do not always align with tokenizer
boundaries because models operate on subword tokens.

Solution:
prefix-based decoding was implemented, allowing partial matches to grow
incrementally until a valid candidate is formed.

---

### Controlled termination of values

Detecting when a generated value should stop was difficult without
breaking structure.

Solution:
generation stops when predefined delimiter tokens are produced,
depending on the expected parameter type.

---

### Hardware variability

The program needed to run across multiple environments without manual
configuration.

Solution:
automatic device detection selects CPU, CUDA, or MPS depending on
availability.

## Testing Strategy

Validation focused on ensuring both structural correctness and expected
behavior.

### Structural validation

Generated outputs were checked to ensure:

- valid JSON parsing
- correct nesting
- presence of required keys.

---

### Functional testing

Different prompt categories were tested:

- arithmetic requests
- string operations
- multiple parameters
- ambiguous phrasing.

---

### Determinism testing

The same inputs were executed multiple times to verify that outputs
remain identical due to greedy decoding.

---

### Error handling tests

Additional tests verified robustness against:

- missing input files
- malformed JSON inputs
- unsupported prompts.

The program must never crash unexpectedly.
# Instructions

### Program runs with following command
```
uv run python -m src [–functions_definition <function_definition_file>]
[–input <input_file>] [–output <output_file>]
```

### But there is a provided Makefile where you can use:
```
make run
```




# Resources
The following materials were used as references during the development of this project:
Documentation

1. Official Python Documentation — https://docs.python.org/3/
Used for understanding standard library modules, typing, and file handling.
2. Python dataclasses documentation — https://docs.python.org/3/library/dataclasses.html
3. Python json module documentation — https://docs.python.org/3/library/json.html

## Articles & Tutorials
1. Real Python — Working with JSON in Python
https://realpython.com/python-json/



## Background information on dataclass design principles.
### AI Usage
Artificial Intelligence tools were used as development assistance during this project.
### Tools
1. Gemini

### How AI Was Used
AI was used as a supportive programming assistant for:
explaining Python concepts and standard library behavior;
suggesting architectural improvements and refactoring ideas;
helping structure parts of the README documentation;
reviewing code logic and identifying potential edge cases;
improving clarity of docstrings and comments.

### How AI Was Not Used
AI did not autonomously generate the entire project.
Core implementation decisions, debugging, and final code integration were performed manually by the author.
All generated suggestions were reviewed, modified, and validated before inclusion