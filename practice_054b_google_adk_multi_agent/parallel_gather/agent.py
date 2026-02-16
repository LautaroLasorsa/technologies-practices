"""Exercise 3: ParallelAgent — Concurrent Gathering.

This exercise demonstrates ADK's ParallelAgent, which runs all its child
agents concurrently — like asyncio.gather(). This is useful when agents
perform independent work (e.g., fetching data from different sources).

Architecture:
    SequentialAgent("parallel_gather_pipeline")
    ├── ParallelAgent("data_gatherers")
    │   ├── TechNewsAgent   — gathers tech news → state["tech_news"]
    │   ├── WeatherAgent    — gathers weather   → state["weather_data"]
    │   └── StockAgent      — gathers stocks    → state["stock_data"]
    └── SummarizerAgent     — reads all 3 keys, produces unified summary

How ParallelAgent works internally:
- It runs ALL sub_agents simultaneously (concurrent async tasks)
- Events from different agents may interleave in the output stream
- All agents share the SAME session state — so each agent MUST write
  to a UNIQUE state key to avoid race conditions
- The ParallelAgent completes when ALL children have completed
- No LLM reasoning for orchestration — it's purely structural

Key concept — avoiding state conflicts:
    Since parallel agents share state, two agents writing to the same key
    creates a race condition. Convention: each agent writes to a unique,
    namespaced key:
        tech_agent  → state["tech_news"]
        weather_agent → state["weather_data"]
        stock_agent → state["stock_data"]

Key concept — ParallelAgent + SequentialAgent composition:
    ParallelAgent only runs agents concurrently — it doesn't merge results.
    To summarize parallel outputs, wrap the ParallelAgent and a Summarizer
    in a SequentialAgent:
        Sequential([Parallel([A, B, C]), Summarizer])
    The Summarizer runs AFTER all parallel agents complete.
"""

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

MODEL = LiteLlm(model="ollama_chat/qwen2.5:7b", api_base="http://localhost:11434")


# ---------------------------------------------------------------------------
# Tools — each writes to a UNIQUE state key (critical for parallel safety)
# ---------------------------------------------------------------------------


def fetch_tech_news(topic: str, tool_context: ToolContext) -> str:
    """Fetch the latest technology news about a topic.

    Simulates an API call to a tech news aggregator. Stores results in
    state["tech_news"] so other agents can read them after parallel execution.

    Args:
        topic: Technology topic to search for.
        tool_context: ADK-injected context for state access.

    Returns:
        Confirmation of fetched articles.
    """
    articles = [
        {"title": f"{topic}: Major Breakthrough Announced", "source": "TechCrunch"},
        {"title": f"How {topic} is Changing the Industry", "source": "Wired"},
        {"title": f"Top 5 {topic} Trends for 2025", "source": "ArsTechnica"},
    ]
    tool_context.state["tech_news"] = articles
    return f"Fetched {len(articles)} tech news articles about '{topic}'."


def fetch_weather(city: str, tool_context: ToolContext) -> str:
    """Fetch current weather data for a city.

    Simulates a weather API call. Stores results in state["weather_data"].

    Args:
        city: City name for weather lookup.
        tool_context: ADK-injected context for state access.

    Returns:
        Current weather summary.
    """
    weather = {
        "city": city,
        "temperature": "22C",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "wind": "12 km/h NW",
    }
    tool_context.state["weather_data"] = weather
    return f"Weather for {city}: {weather['temperature']}, {weather['condition']}."


def fetch_stock_data(symbol: str, tool_context: ToolContext) -> str:
    """Fetch stock market data for a given ticker symbol.

    Simulates a financial data API call. Stores results in state["stock_data"].

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL").
        tool_context: ADK-injected context for state access.

    Returns:
        Summary of stock data.
    """
    stock = {
        "symbol": symbol.upper(),
        "price": 182.52,
        "change": "+2.34%",
        "volume": "45.2M",
        "market_cap": "2.8T",
    }
    tool_context.state["stock_data"] = stock
    return f"Stock {stock['symbol']}: ${stock['price']} ({stock['change']})."


def read_all_gathered_data(tool_context: ToolContext) -> str:
    """Read all data gathered by the parallel agents from session state.

    This tool is used by the summarizer agent AFTER the ParallelAgent
    completes. It reads from the three state keys written by the parallel
    agents and returns a consolidated view.

    Args:
        tool_context: ADK-injected context for state access.

    Returns:
        A formatted string of all gathered data.
    """
    tech = tool_context.state.get("tech_news", "No tech news gathered.")
    weather = tool_context.state.get("weather_data", "No weather data gathered.")
    stock = tool_context.state.get("stock_data", "No stock data gathered.")

    return (
        f"=== Gathered Data Summary ===\n"
        f"Tech News: {tech}\n"
        f"Weather: {weather}\n"
        f"Stock Data: {stock}\n"
        f"============================="
    )


# ---------------------------------------------------------------------------
# TODO(human): Create the parallel gathering agents and the combined pipeline.
# ---------------------------------------------------------------------------
#
# Your task is to create 3 data-gathering agents, wrap them in a
# ParallelAgent, then combine with a summarizer in a SequentialAgent.
#
# Step 1 — Create the three gathering agents
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create three LlmAgent instances, one per data source:
#
#   tech_news_agent = LlmAgent(
#       name="tech_news_agent",
#       model=MODEL,
#       instruction: Tell it to fetch tech news about the user's topic
#           using the `fetch_tech_news` tool. It should call the tool
#           and report what it found.
#       description: "Fetches the latest technology news articles."
#       tools: [fetch_tech_news],
#       output_key: "tech_news_output",
#   )
#
#   weather_agent = LlmAgent(
#       name="weather_agent",
#       model=MODEL,
#       instruction: Tell it to fetch weather data for a relevant city
#           (it can pick a default like "San Francisco" or use context).
#           It should call `fetch_weather` and report the conditions.
#       description: "Fetches current weather information."
#       tools: [fetch_weather],
#       output_key: "weather_output",
#   )
#
#   stock_agent = LlmAgent(
#       name="stock_agent",
#       model=MODEL,
#       instruction: Tell it to fetch stock data for a relevant ticker
#           (it can pick a default like "GOOGL" or use context).
#           It should call `fetch_stock_data` and report the price.
#       description: "Fetches stock market data."
#       tools: [fetch_stock_data],
#       output_key: "stock_output",
#   )
#
# IMPORTANT: Each agent writes to a UNIQUE state key via its tool.
# This is critical because they run concurrently — writing to the same
# key would create a race condition.
#
# Step 2 — Create the ParallelAgent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wrap the three gathering agents in a ParallelAgent:
#
#   parallel_gatherer = ParallelAgent(
#       name="parallel_gatherer",
#       sub_agents=[tech_news_agent, weather_agent, stock_agent],
#       description="Gathers data from multiple sources concurrently.",
#   )
#
# Note: ParallelAgent does NOT take a `model` parameter. It's a workflow
# agent — no LLM reasoning, just concurrent execution of children.
#
# Step 3 — Create the Summarizer agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create an LlmAgent that reads all gathered data and produces a summary:
#
#   summarizer_agent = LlmAgent(
#       name="summarizer_agent",
#       model=MODEL,
#       instruction: Tell it to call `read_all_gathered_data` to get the
#           data collected by the parallel agents, then produce a unified
#           "daily briefing" style summary combining tech news, weather,
#           and stock info.
#       description: "Summarizes all gathered data into a unified briefing."
#       tools: [read_all_gathered_data],
#       output_key: "final_summary",
#   )
#
# Step 4 — Combine into a SequentialAgent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The final pipeline: first gather in parallel, then summarize.
#
#   root_agent = SequentialAgent(
#       name="parallel_gather_pipeline",
#       sub_agents=[parallel_gatherer, summarizer_agent],
#       description="Gathers data in parallel, then summarizes.",
#   )
#
# This creates the architecture:
#   Sequential
#   ├── Parallel (all 3 run concurrently)
#   │   ├── tech_news_agent
#   │   ├── weather_agent
#   │   └── stock_agent
#   └── summarizer_agent (runs after all parallel agents complete)
#
# Test prompts to try:
#   - "Give me a daily briefing about AI"
#   - "What's the latest on quantum computing?"
# Watch the parallel agents execute simultaneously, then the summarizer
# reads all their outputs from state.
# ---------------------------------------------------------------------------

raise NotImplementedError(
    "TODO(human): Create tech_news_agent, weather_agent, stock_agent, "
    "parallel_gatherer (ParallelAgent), summarizer_agent, and the "
    "root_agent (SequentialAgent wrapping parallel + summarizer)."
)
