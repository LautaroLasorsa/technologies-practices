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

### Phase 1: Setup & First Signature (~15 min)

Start Docker and pull the Ollama model. Verify DSPy can connect to the local Ollama instance.

**Exercise 1 — Configure DSPy LM** (`src/01_first_signature.py`, TODO #1):
DSPy needs a configured language model backend before any module can run. This exercise teaches the fundamental setup: pointing DSPy at a local Ollama instance. Understanding this configuration is essential because DSPy abstracts the LM — the same program works with OpenAI, Anthropic, or local models just by changing this one line.

**Exercise 2 — First String Signature** (`src/01_first_signature.py`, TODO #2):
String signatures are DSPy's simplest abstraction — they replace prompt templates with declarative I/O specs. By testing with `dspy.Predict`, you see how DSPy converts your signature into an actual prompt behind the scenes. This is the foundation everything else builds on.

### Phase 2: Modules — Predict, CoT, PoT (~20 min)

Explore the three built-in module strategies and understand when each is appropriate.

**Exercise 3 — Compare Module Strategies** (`src/02_modules.py`, TODO #1):
Running the same task through Predict, ChainOfThought, and ProgramOfThought reveals how each module transforms the underlying prompt differently. Predict gives a direct answer, CoT forces reasoning before answering, and PoT generates executable code. Seeing the differences firsthand builds intuition for which strategy to choose.

**Exercise 4 — Class-Based Signature** (`src/02_modules.py`, TODO #2):
Class-based signatures are DSPy's production-grade way to define tasks. Typed fields with descriptions give the optimizer more information to work with during compilation. The docstring becomes the task instruction, making your code self-documenting.

### Phase 3: Custom Modules (~20 min)

Learn the Module composition pattern — DSPy's core design for building multi-step programs.

**Exercise 5 — SentimentAnalyzer Module** (`src/03_custom_module.py`, TODO):
Building a custom module teaches the most important DSPy pattern: composing simple modules into complex programs. The `__init__` + `forward()` pattern mirrors PyTorch's `nn.Module`, and for good reason — each sub-module is independently optimizable. The optimizer can find different few-shot examples for each step.

### Phase 4: Metrics & Datasets (~15 min)

Define evaluation criteria and prepare data for optimization.

**Exercise 6 — Metric Function** (`src/04_metrics_datasets.py`, TODO #1):
Metrics are the compass that guides DSPy's optimization. Without a good metric, the optimizer cannot distinguish good outputs from bad. This exercise teaches the metric function signature and how to design metrics that capture your task's success criteria.

**Exercise 7 — Train/Val Datasets** (`src/04_metrics_datasets.py`, TODO #2):
DSPy.Example objects are the standard data container. The `.with_inputs()` method is critical — it tells the optimizer which fields are inputs (available at inference) vs labels (used only for evaluation). Getting this wrong means the optimizer "cheats" by including labels in the prompt.

### Phase 5: BootstrapFewShot Optimization (~20 min)

Run the full optimization loop and compare results.

**Exercise 8 — Compile with BootstrapFewShot** (`src/05_optimization.py`, TODO #1):
This is where everything comes together. Compilation runs the teacher on training data, filters traces with your metric, and injects the best demonstrations into the student program. Understanding the `max_bootstrapped_demos` and `max_labeled_demos` parameters is key to controlling optimization.

**Exercise 9 — Evaluate & Compare** (`src/05_optimization.py`, TODO #2):
Running `dspy.Evaluate` on both baseline and optimized programs provides concrete evidence that optimization works. Inspecting the optimized program's demos reveals *what* the optimizer learned — which examples it selected and why they help.

## Motivation

DSPy represents a paradigm shift in LLM development — from manual prompt engineering to systematic optimization. Understanding this approach is increasingly important as LLM applications move from prototypes to production, where reproducibility and measurable improvement matter. Foundation for advanced DSPy (030b) and LangGraph integration (030c).

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container |
| | `docker exec ollama ollama pull qwen2.5:7b` | Download model for exercises |
| **Setup** | `uv sync` | Install Python dependencies |
| | `uv run python src/00_verify_setup.py` | Verify DSPy + Ollama connection |
| **Phase 1** | `uv run python src/01_first_signature.py` | Test first signature with Predict |
| **Phase 2** | `uv run python src/02_modules.py` | Compare Predict vs CoT vs PoT |
| **Phase 3** | `uv run python src/03_custom_module.py` | Run custom module composition |
| **Phase 4** | `uv run python src/04_metrics_datasets.py` | Test metrics and dataset creation |
| **Phase 5** | `uv run python src/05_optimization.py` | Run BootstrapFewShot optimization |

## References

- DSPy Official Docs: https://dspy.ai/
- DSPy Signatures: https://dspy.ai/learn/programming/signatures/
- DSPy Modules: https://dspy.ai/learn/programming/modules/
- DSPy Optimizers: https://dspy.ai/learn/optimization/optimizers/
- BootstrapFewShot API: https://dspy.ai/api/optimizers/BootstrapFewShot/
