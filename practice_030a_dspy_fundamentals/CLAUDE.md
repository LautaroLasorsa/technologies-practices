# Practice 030a — DSPy Fundamentals: Signatures, Modules & Optimization

## Technologies

- **DSPy (v2.5+)**: Declarative framework for programming (not prompting) language models
- **Signatures**: Declarative input/output specifications replacing manual prompts
- **Modules**: Composable LM components (Predict, ChainOfThought, ProgramOfThought)
- **Optimizers**: Algorithms that compile programs into optimized prompts (BootstrapFewShot)

## Stack

Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### What DSPy Is & the Problem It Solves

DSPy (Declarative Self-improving Python) shifts LLM development from manual prompt engineering to **programmatic optimization**. Instead of tweaking prompt strings by hand — adjusting wording, adding examples, rearranging instructions — you write modular Python code that specifies *what* the LLM should do (via signatures), compose those specifications into programs (via modules), define a metric that measures success, and let an optimizer find the best prompts and few-shot examples automatically.

**Analogy**: DSPy is to prompt engineering what PyTorch is to manual gradient computation. In PyTorch you define the model architecture and loss function, then autograd + optimizer handle backpropagation. In DSPy you define the program structure and metric, then a teleprompter handles prompt optimization. Both automate the tedious, error-prone optimization loop.

**Why this matters**: Manual prompt engineering is fragile — prompts that work for GPT-4 break on Claude, prompts tuned for one data distribution fail on another, and there's no systematic way to improve. DSPy makes LLM programs **portable** (change the model, re-optimize), **measurable** (metric-driven evaluation), and **reproducible** (deterministic compilation).

### Core Paradigm — Three Abstractions

#### 1. Signatures

Declarative I/O specifications that replace manual prompt templates. They tell DSPy *what* transformation the LM should perform, without specifying *how*.

**String format** (quick prototyping):
```python
"question -> answer"
"context, question -> answer"
"document -> summary, key_points"
```

**Class format** (production, typed):
```python
class QA(dspy.Signature):
    """Answer the question based on the given context."""
    context = dspy.InputField(desc="relevant facts")
    question = dspy.InputField(desc="user question")
    answer = dspy.OutputField(desc="concise answer")
```

The signature's **docstring becomes the task description** in the generated prompt. Field descriptions guide the LM's behavior. This is not just syntactic sugar — the optimizer uses these descriptions when generating and selecting demonstrations.

#### 2. Modules

Composable components that wrap signatures with prompting strategies. Each module takes a signature and decides *how* to prompt the LM.

- **`dspy.Predict`**: Direct LM call. Simplest module — takes input fields, produces output fields. No added reasoning. Best for simple, well-defined tasks.
- **`dspy.ChainOfThought`**: Adds an intermediate `reasoning` field before the output. The LM "thinks step by step" before answering. Best for tasks requiring multi-step logic.
- **`dspy.ProgramOfThought`**: Generates Python code to compute the answer, then executes it. Eliminates LM arithmetic/logic errors. Best for math, data manipulation, or any task where code execution beats LM reasoning.

**Custom modules**: Subclass `dspy.Module`, define `__init__` (declare sub-modules) and `forward()` (define logic). This is how you compose multi-step programs:

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)
```

#### 3. Optimizers (Teleprompters)

Algorithms that compile programs into optimized versions by finding the best prompts and few-shot demonstrations.

- **`BootstrapFewShot`**: Uses a teacher LM to generate example traces, filters them with a metric, and selects the best as few-shot demonstrations for the student program. Simple, effective, works with small datasets (10-50 examples).
- **`MIPROv2`**: Bayesian optimization over instructions + demonstrations. More powerful but requires more data (50-500+ examples) and compute.

The result of optimization is a compiled program — same code, but with optimized prompts baked in.

### How Optimization Works (BootstrapFewShot)

1. **Teacher runs**: The teacher LM (can be the same model or a stronger one) runs the program on each training example.
2. **Metric filters**: A metric function evaluates each trace — did the program produce a correct output?
3. **Demo selection**: Traces that pass the metric threshold become few-shot demonstrations.
4. **Student receives demos**: The student program's prompt now includes these curated examples.
5. **Result**: The student performs better because it has high-quality, task-specific demonstrations.

This is essentially **automated few-shot example curation** — the optimizer finds examples that best teach the model how to perform your specific task.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Signature** | Declarative I/O spec (`"context, question -> answer"`). Replaces prompt templates. |
| **InputField / OutputField** | Typed fields with descriptions. Guide the LM's behavior. |
| **dspy.Predict** | Simplest module — direct LM call with no added reasoning. |
| **dspy.ChainOfThought** | Adds intermediate reasoning step before output. Best for complex tasks. |
| **dspy.ProgramOfThought** | Generates Python code for execution. Eliminates LM computation errors. |
| **dspy.Module** | Base class for custom modules. Define `__init__` + `forward()`. |
| **dspy.Example** | Data container with input/output fields. Used for train/val datasets. |
| **Metric** | Python function `(example, prediction, trace) -> score`. Guides optimization. |
| **Optimizer** | Algorithm that compiles a program (BootstrapFewShot, MIPROv2, etc.). |
| **Teleprompter** | Legacy name for optimizer. |
| **Compilation** | Process of running optimizer on program + dataset to produce an optimized program. |

### Data Requirements

- **Train set**: 10-50 examples for BootstrapFewShot, 50-500+ for MIPROv2.
- **Important**: Use a 20/80 train/val split for prompt optimizers (they overfit easily to the training set).
- **Example creation**: `dspy.Example(input_field=..., output_field=...).with_inputs("input_field")` — the `.with_inputs()` call marks which fields are inputs (the rest are labels).

### Ecosystem Context

| Comparison | Key Difference |
|-----------|---------------|
| **DSPy vs LangChain** | LangChain = orchestration framework (chains, tools, memory). DSPy = optimization framework (automated prompt tuning). Complementary, not competing. |
| **DSPy vs manual prompting** | DSPy automates what prompt engineers do manually, but requires defining metrics and having training data. |
| **When DSPy shines** | You have a measurable metric, some labeled examples, and want reproducible, improvable results. |
| **When to skip DSPy** | One-off creative tasks, no metric available, no labeled data. |

## Description

Hands-on introduction to DSPy's programming paradigm. You'll define signatures, build modules (Predict, ChainOfThought, ProgramOfThought), compose custom modules, create metrics, and run BootstrapFewShot optimization to see DSPy automatically improve your program's performance.

**What you'll learn:**
- DSPy's "program, don't prompt" paradigm
- Signature definition (string and class formats)
- Built-in modules and when to use each
- Custom module composition (subclass dspy.Module)
- Metric design for automated evaluation
- BootstrapFewShot optimization workflow
- Comparing baseline vs optimized program performance

## Instructions

### Phase 1: Setup & First Signature (~15 min) — `src/_01_first_signature.py`

Start Docker and pull the Ollama model, then run `_00_verify_setup.py` to confirm the connection.

1. **TODO #1 — `configure_dspy()`**: build a `dspy.LM` pointed at local Ollama (`ollama_chat/qwen2.5:7b`, `api_base="http://localhost:11434"`, empty `api_key`) and call `dspy.configure(lm=...)`. Teaches DSPy's global LM configuration — the same program works with OpenAI, Anthropic, or local models just by swapping this one line.
2. **TODO #2 — `build_qa_predictor()`**: return a `dspy.Predict("question -> answer")`. The scaffolded demo loop exercises it on three questions and prints the parsed `result.answer`, showing how DSPy turns a string signature into a prompt and back into typed fields.

### Phase 2: Modules — Predict, CoT, PoT (~20 min) — `src/_02_modules.py`

Compare the three built-in module strategies, then graduate to a class-based signature.

3. **TODO #1 — `build_predict_module()`**: return `dspy.Predict(MathProblem)`. Direct LM call, no reasoning trace — the baseline against which CoT/PoT will be measured.
4. **TODO #2 — `build_cot_module()`**: return `dspy.ChainOfThought(MathProblem)`. Adds an intermediate `reasoning` field; the demo prints it so you can see the model "think out loud".
5. **TODO #3 — `build_pot_module()`**: return `dspy.ProgramOfThought(MathProblem)`. The LM writes Python code, DSPy executes it — eliminates arithmetic errors entirely.
6. **TODO #4 — `TextClassification` class signature**: declare a `dspy.Signature` subclass with the docstring task instruction and four typed fields (`text`, `categories` inputs; `category`, `confidence` outputs). Class signatures expose the field descriptions and docstring to the optimizer at compile time.
7. **TODO #5 — `build_classifier()`**: return `dspy.ChainOfThought(TextClassification)`. Pairing the typed signature with CoT lets the LM justify its category choice before committing.

### Phase 3: Custom Modules (~20 min) — `src/_03_custom_module.py`

Subclass `dspy.Module` to chain two sub-modules into a sentiment pipeline.

8. **TODO #1 — `SentimentAnalyzer.__init__`**: declare `self.extract` (CoT for `review -> aspects: list[str]`) and `self.analyze` (CoT for `review, aspects -> sentiment, confidence`). Storing each sub-module as an attribute is what makes the optimizer able to find and tune them independently.
9. **TODO #2 — `SentimentAnalyzer.forward`**: call `self.extract`, then `self.analyze`, then bundle the four interesting fields into a `dspy.Prediction`. This is the PyTorch-style composition pattern that powers every multi-step DSPy program.

### Phase 4: Metrics & Datasets (~15 min) — `src/_04_metrics_datasets.py`

Define the evaluation criterion and prepare labeled data for optimization.

10. **TODO #1 — `sentiment_metric(example, pred, trace=None)`**: return `1.0` if the lowercased/stripped predicted sentiment matches the example's gold label, else `0.0`. Metrics are the optimizer's only signal — without them, BootstrapFewShot cannot distinguish good demonstrations from bad ones.
11. **TODO #2 — `create_datasets()`**: turn `RAW_DATA` into a list of `dspy.Example` objects (each one calling `.with_inputs("review")` so `sentiment` is treated as a label, not a model input), then split into train (`TRAIN_SIZE`) and validation. Forgetting `.with_inputs(...)` lets the optimizer "cheat" by feeding the label into the prompt.

### Phase 5: BootstrapFewShot Optimization (~20 min) — `src/_05_optimization.py`

Compile, evaluate, and inspect what the optimizer learned. Paste the Phase 3/4 stubs from your earlier work into the marked spots before starting these TODOs.

12. **TODO #1 — `compile_program(train_set)`**: build a `BootstrapFewShot(metric=sentiment_metric, max_bootstrapped_demos=4, max_labeled_demos=8)`, then return both a fresh baseline `SentimentAnalyzer()` and `optimizer.compile(student=SentimentAnalyzer(), trainset=train_set)`. Watch the console — DSPy logs which traces pass the metric and become demos.
13. **TODO #2 — `score_programs(baseline, optimized, val_set)`**: build a `dspy.Evaluate(devset=val_set, metric=sentiment_metric, num_threads=1, display_progress=True)` and return `(evaluator(baseline), evaluator(optimized))`. The scaffolded reporter prints both scores and the delta.
14. **TODO #3 — `print_optimized_demos(optimized)`**: iterate `optimized.named_predictors()` and print each predictor's `len(predictor.demos)` and the demos themselves. This exposes the auto-curated few-shot examples that explain *why* the optimized program does better.

## Motivation

DSPy represents a paradigm shift in LLM development — from manual prompt engineering to systematic optimization. Understanding this approach is increasingly important as LLM applications move from prototypes to production, where reproducibility and measurable improvement matter. Foundation for advanced DSPy (030b) and LangGraph integration (030c).

## LLM Configuration

By default the practice runs against local Ollama (`qwen2.5:7b`). To switch providers, copy `.env.example` to `.env` and set the variables below before running any script.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `lmstudio` \| `openai` \| `anthropic` \| `google` |
| `LLM_MODEL` | `qwen2.5:7b` | Model name without the provider prefix |
| `LLM_BASE_URL` | _(provider default)_ | Override the API base URL |
| `LLM_API_KEY` | _(empty)_ | API key — required for cloud providers |

All provider routing is centralised in `src/llm_config.py` (`get_lm()` / `configure_lm()`). The `_01_first_signature.py` TODO(human) exercise still asks you to construct a `dspy.LM` manually — that's intentional, as the exercise teaches the raw API before the abstraction is used everywhere else.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container |
| | `docker exec ollama ollama pull qwen2.5:7b` | Download model for exercises |
| **Setup** | `uv sync` | Install Python dependencies |
| | `uv run python -m src._00_verify_setup` | Verify DSPy + Ollama connection |
| **Phase 1** | `uv run python -m src._01_first_signature` | Configure DSPy + run a string signature |
| **Phase 2** | `uv run python -m src._02_modules` | Compare Predict vs CoT vs PoT, then class signature |
| **Phase 3** | `uv run python -m src._03_custom_module` | Run the custom `SentimentAnalyzer` module |
| **Phase 4** | `uv run python -m src._04_metrics_datasets` | Build datasets and sanity-check the metric |
| **Phase 5** | `uv run python -m src._05_optimization` | Compile with BootstrapFewShot + evaluate |
| **Cleanup** | `python clean.py` | Remove caches, venv, Docker volumes |

## References

- DSPy Official Docs: https://dspy.ai/
- DSPy Signatures: https://dspy.ai/learn/programming/signatures/
- DSPy Modules: https://dspy.ai/learn/programming/modules/
- DSPy Optimizers: https://dspy.ai/learn/optimization/optimizers/
- BootstrapFewShot API: https://dspy.ai/api/optimizers/BootstrapFewShot/
