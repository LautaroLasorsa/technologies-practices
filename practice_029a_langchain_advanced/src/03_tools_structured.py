"""
Practice 029a — Phase 3: Custom Tools & Structured Output

This exercise teaches three critical production patterns:
  1. Creating tools that LLMs can discover and call via schemas
  2. Extracting structured data from free-form LLM output
  3. Building resilient chains with fallback strategies
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Setup: model and pre-built schemas
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

parser = StrOutputParser()


# --- Pydantic model for structured output (Exercise 2) ---

class Recipe(BaseModel):
    """A structured recipe extracted from unstructured cooking text."""
    title: str = Field(description="Name of the recipe")
    servings: int = Field(description="Number of servings")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    ingredients: list[str] = Field(description="List of ingredients with quantities")
    steps: list[str] = Field(description="Ordered list of cooking steps")
    difficulty: str = Field(description="Easy, Medium, or Hard")


# Sample unstructured cooking text for structured extraction.
RECIPE_TEXT = """
Last weekend I made the most amazing garlic butter pasta! It's super simple
and feeds about 4 people. I started by boiling a pound of spaghetti in salted
water — that takes about 10 minutes. While the pasta cooked, I minced 6 cloves
of garlic and melted half a stick of butter (that's 4 tablespoons) in a large
skillet over medium heat. Once the butter was foamy, I tossed in the garlic
and cooked it for about a minute until fragrant. Then I added a quarter cup of
olive oil, a pinch of red pepper flakes, and some salt. When the pasta was al
dente, I drained it (saving a cup of pasta water!) and tossed it right into the
skillet. I added about half the pasta water and a big handful of freshly grated
Parmesan — maybe a cup? Tossed everything together until it was glossy and
coated. Finished with fresh parsley and more Parmesan on top. Honestly, the
whole thing took maybe 20 minutes from start to finish. Anyone can make this!
"""


# ---------------------------------------------------------------------------
# Exercise 1: Create custom tools
# ---------------------------------------------------------------------------

def exercise_custom_tools() -> None:
    # TODO(human): Create 3 custom tools using the @tool decorator.
    #
    # WHAT TO DO:
    #   Define three functions decorated with @tool. Each must have:
    #     - Type-annotated parameters (LangChain auto-generates the JSON schema)
    #     - A descriptive docstring (the LLM reads this to decide when to use the tool)
    #     - A return value
    #
    #   Tool 1 — Calculator:
    #     @tool
    #     def calculator(expression: str) -> str:
    #         """Evaluate a mathematical expression and return the result."""
    #         ...  # Use eval() for simplicity, or parse manually
    #
    #   Tool 2 — Word counter:
    #     @tool
    #     def word_counter(text: str) -> str:
    #         """Count the number of words in the given text."""
    #         ...
    #
    #   Tool 3 — Text reverser:
    #     @tool
    #     def text_reverser(text: str) -> str:
    #         """Reverse the given text character by character."""
    #         ...
    #
    #   After defining the tools, print their metadata to see what the LLM sees:
    #     for t in [calculator, word_counter, text_reverser]:
    #         print(f"Name: {t.name}")
    #         print(f"Description: {t.description}")
    #         print(f"Schema: {t.args_schema.model_json_schema()}")
    #         print()
    #
    #   Then invoke each tool directly to verify they work:
    #     print(calculator.invoke({"expression": "2 ** 10"}))
    #     print(word_counter.invoke({"text": "hello world foo bar"}))
    #     print(text_reverser.invoke({"text": "LangChain"}))
    #
    # WHY THIS MATTERS:
    #   Tools are how LLMs interact with the outside world. The @tool decorator
    #   automatically generates a JSON schema from your function's type hints
    #   and docstring. This schema is what the LLM sees when deciding which
    #   tool to call and with what arguments. Good type hints + descriptive
    #   docstrings = better tool selection by the LLM.
    #
    #   In LangChain, tools are Runnables — you can invoke them, batch them,
    #   and compose them into chains just like any other component.
    #
    # EXPECTED BEHAVIOR:
    #   You should see the name, description, and JSON schema for each tool,
    #   followed by the results of invoking each one directly.
    raise NotImplementedError("Create custom tools here")


# ---------------------------------------------------------------------------
# Exercise 2: Structured output with .with_structured_output()
# ---------------------------------------------------------------------------

def exercise_structured_output() -> None:
    # TODO(human): Extract a structured Recipe from unstructured cooking text.
    #
    # WHAT TO DO:
    #   1. Create a structured LLM by calling:
    #        structured_llm = llm.with_structured_output(Recipe)
    #      This wraps the model to return a validated Recipe instance instead
    #      of raw text. Under the hood, it uses Ollama's JSON schema support
    #      to constrain the output format.
    #
    #   2. Create a prompt that instructs the LLM to extract recipe info:
    #        prompt = ChatPromptTemplate.from_messages([
    #            ("system", "Extract the recipe details from the text. "
    #                       "Return structured data matching the schema."),
    #            ("human", "{text}"),
    #        ])
    #
    #   3. Build the chain:
    #        chain = prompt | structured_llm
    #      Note: no output parser needed! with_structured_output() already
    #      returns a Pydantic instance (or dict, depending on config).
    #
    #   4. Invoke with {"text": RECIPE_TEXT} and print the result.
    #      Access individual fields: result.title, result.ingredients, etc.
    #
    # WHY THIS MATTERS:
    #   This is the MOST IMPORTANT production pattern in LangChain. LLMs
    #   produce free-form text, but applications need structured data. The
    #   .with_structured_output() method solves this by:
    #     - Sending the Pydantic schema to the model as a JSON schema
    #     - Parsing and validating the model's JSON output
    #     - Returning a typed Pydantic instance you can use in code
    #
    #   Without this, you'd be writing fragile regex parsers or hoping the
    #   LLM follows format instructions in the prompt. with_structured_output()
    #   makes structured extraction reliable and type-safe.
    #
    # EXPECTED BEHAVIOR:
    #   A Recipe object with title="Garlic Butter Pasta" (or similar),
    #   servings=4, prep_time_minutes=20, a list of ingredients, ordered
    #   steps, and difficulty="Easy". All fields populated from the text.
    raise NotImplementedError("Extract structured recipe here")


# ---------------------------------------------------------------------------
# Exercise 3: Fallback chain
# ---------------------------------------------------------------------------

def exercise_fallback_chain() -> None:
    # TODO(human): Build a chain with automatic fallback on failure.
    #
    # WHAT TO DO:
    #   1. Create a "primary" chain that uses structured output:
    #        structured_llm = llm.with_structured_output(Recipe)
    #        primary_prompt = ChatPromptTemplate.from_messages([
    #            ("system", "Extract recipe details as structured data."),
    #            ("human", "{text}"),
    #        ])
    #        primary_chain = primary_prompt | structured_llm
    #
    #   2. Create a "fallback" chain that uses plain text extraction:
    #        fallback_prompt = ChatPromptTemplate.from_messages([
    #            ("system", "Extract the recipe title, ingredients, and steps "
    #                       "from the text. Format as readable text."),
    #            ("human", "{text}"),
    #        ])
    #        fallback_chain = fallback_prompt | llm | parser
    #
    #   3. Combine them using .with_fallbacks():
    #        resilient_chain = primary_chain.with_fallbacks([fallback_chain])
    #
    #   4. Test with RECIPE_TEXT (should use primary). Then test with
    #      deliberately bad input (e.g., "This is not a recipe at all,
    #      just random text about weather.") to see the fallback activate.
    #      Print which chain was used and the result.
    #
    # WHY THIS MATTERS:
    #   LLMs are non-deterministic — structured output can fail (invalid JSON,
    #   missing fields, validation errors). In production, you need graceful
    #   degradation. The .with_fallbacks() method automatically retries with
    #   the next chain in the list when the current one raises an exception.
    #
    #   This pattern is essential for reliability: structured extraction is
    #   preferred (type-safe, easy to use), but plain text extraction is
    #   the safety net that always works. The fallback activates transparently
    #   — calling code doesn't need to handle the retry logic.
    #
    # EXPECTED BEHAVIOR:
    #   For RECIPE_TEXT: primary chain succeeds, returns a Recipe object.
    #   For bad input: primary may fail validation, fallback returns text.
    #   (Note: with a capable model like qwen2.5:7b, the primary might
    #   succeed even on edge cases — that's fine. The pattern is what matters.)
    raise NotImplementedError("Build fallback chain here")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: Custom Tools & Structured Output")
    print("=" * 60)

    print("\n--- Exercise 1: Custom tools ---\n")
    exercise_custom_tools()

    print("\n--- Exercise 2: Structured output ---\n")
    exercise_structured_output()

    print("\n--- Exercise 3: Fallback chain ---\n")
    exercise_fallback_chain()
