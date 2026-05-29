# Baselines/React/react_baseline.py
import re
import os
import json
import yaml
import requests
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add
from bs4 import BeautifulSoup

# LangGraph Core Orchestration
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Tools Stack
from ddgs import DDGS
from docling.document_converter import DocumentConverter

# ENVIRONMENT CONFIGURATION LOADER
current_script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(current_script_dir, "config.yaml")

if os.path.exists(yaml_config_path):
    with open(yaml_config_path, "r", encoding="utf-8") as y_f:
        CONFIG = yaml.safe_load(y_f)
else:
    # Fail-safe backup defaults to prevent LangGraph fallback to minimum budget caps
    CONFIG = {
        "llm": {"model_name": "gpt-4o", "temperature": 0.0},
        "tools": {"search_max_results": 5},
        "runtime": {"max_iterations": 20},
        "paths": {"prompt_file": "prompt_template.txt"}
    }

# Enforce explicit integer validation for budget conditions
MAX_ITERS = int(CONFIG["runtime"]["max_iterations"])

# ----------------------------------------------------
# A. SYSTEM & AGENT STATE
# ----------------------------------------------------
class ReActState(TypedDict):
    trajectory: List[BaseMessage]
    task_query: str
    loop_count: int
    extracted_table: Annotated[List[Dict[str, Any]], add]


# B. PRIMITIVE TOOLS
def run_dux_global_search(query: str, max_results: int) -> str:
    """Uses Dux Distributed Global Search (DDGS) to query across web services."""
    print(f"\n[SEARCH ENGINE] Querying: '{query}'")
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No search results found from Dux Distributed Search backends."

        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n---")
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Dux Metasearch runtime failure: {str(e)}"


def run_bs4_docling_extractor(url: str) -> List[Dict[str, Any]]:
    """Fetches HTML via Requests, parses with BeautifulSoup, flattens via Docling, and extracts via Pydantic + LLM."""
    print(f"\n[REQUESTS & BS4] Fetching HTML from source URL: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        cleaned_html_str = str(soup)
        temp_html_path = os.path.join(current_script_dir, f"temp_bs4_{os.getpid()}.html")
        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(cleaned_html_str)

        print("[DOCLING CONVERTER] Generating layout-preserved structural markdown...")
        converter = DocumentConverter()
        docling_result = converter.convert(temp_html_path)
        markdown_text = docling_result.document.export_to_markdown()
        os.remove(temp_html_path)

        print("[PYDANTIC EXTRACTOR] Extracting structure rows using schema...")
        from schema import EntityTable
        extractor_llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
        structured_llm = extractor_llm.with_structured_output(EntityTable)

        extraction_prompt = (
            f"Extract all public health and community facility rows that fit the schema fields from this text:\n\n"
            f"Document Markdown Content:\n{markdown_text}"
        )

        extracted_pydantic_table = structured_llm.invoke(extraction_prompt)
        return [row.model_dump() for row in extracted_pydantic_table.rows]
    except Exception as e:
        print(f"[TOOL EXCEPTION] Pipeline processing failure on target URL {url}: {str(e)}")
        return []



# C. LANGGRAPH EXECUTION ARCHITECTURE NODES
def call_llm_agent(state: ReActState) -> dict:
    current_count = state.get("loop_count", 0) + 1
    prompt_path = os.path.join(current_script_dir, CONFIG["paths"]["prompt_file"])

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_skeleton = f.read()

    formatted_prompt = prompt_skeleton.format(task_query=state["task_query"])
    messages = [HumanMessage(content=formatted_prompt)] + state["trajectory"]

    llm = ChatOpenAI(model=CONFIG["llm"]["model_name"], temperature=CONFIG["llm"]["temperature"])
    response = llm.invoke(messages)

    return {
        "trajectory": [response],
        "loop_count": current_count
    }


def execute_action(state: ReActState) -> dict:
    last_message = state["trajectory"][-1].content
    observation_text = ""
    new_rows = []

    tool_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", last_message)

    if tool_match:
        tool_name, tool_arg = tool_match.groups()
        tool_arg = tool_arg.strip()

        if tool_name == "Search":
            observation_text = run_dux_global_search(
                query=tool_arg,
                max_results=CONFIG["tools"]["search_max_results"]
            )
        elif tool_name == "BrowseAndExtract":
            new_rows = run_bs4_docling_extractor(url=tool_arg)
            observation_text = json.dumps(new_rows, indent=2)
        else:
            observation_text = f"Error: Tool name '{tool_name}' is not recognized."
    else:
        observation_text = "Syntax Error: Must explicitly declare formatting pattern as 'Action: Tool[arg]'."

    print(f"\n[OBSERVATION AT DEPTH {state['loop_count']}]: Captured {len(new_rows)} structured records.\n")
    return {
        "trajectory": [HumanMessage(content=f"Observation: {observation_text}")],
        "extracted_table": new_rows
    }


def router_condition(state: ReActState) -> str:
    last_message = state["trajectory"][-1].content
    if state["loop_count"] >= MAX_ITERS:
        print(f"\n[BUDGET SHUTDOWN] Reached iteration limit threshold of {MAX_ITERS}.")
        return "end"
    if "Final Answer:" in last_message:
        return "end"
    if "Action:" in last_message and "PAUSE" in last_message:
        return "execute"
    return "end"


def compile_react_baseline():
    workflow = StateGraph(ReActState)
    workflow.add_node("agent", call_llm_agent)
    workflow.add_node("tools", execute_action)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router_condition, {"execute": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()
