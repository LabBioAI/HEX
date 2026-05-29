# Baselines/HtmlRag/html_rag.py
import os
import yaml
import requests
from typing import TypedDict, List, Dict, Any
from bs4 import BeautifulSoup
from prompts.schema import EntityTable
# LangGraph Core Architecture Elements
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# Baseline Primitives
from ddgs import DDGS
from docling.document_converter import DocumentConverter


# YAML PARAMETERS CONFIG LOADER
current_script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(current_script_dir, "config.yaml")

if os.path.exists(yaml_config_path):
    with open(yaml_config_path, "r", encoding="utf-8") as y_f:
        CONFIG = yaml.safe_load(y_f)
else:
    CONFIG = {
        "llm": {"model_name": "gpt-4o", "temperature": 0.0},
        "tools": {"search_max_results": 5},
        "paths": {"prompt_file": "html_rag_prompt.txt"}
    }


# A. STATE DEFINITION
class HtmlRagState(TypedDict):
    task_query: str
    discovered_urls: List[str]
    raw_markdown_corpus: str  # Tracks aggregated clean markdown text
    extracted_table: List[Dict[str, Any]]


# B. OPERATIONAL NODES
def execute_discovery_search(state: HtmlRagState) -> dict:
    """Discovers web resources using Dux Distributed Global Search (DDGS)."""
    query = state["task_query"]
    max_results = int(CONFIG["tools"]["search_max_results"])
    print(f"\n[HTML-RAG DISCOVERY] Issuing search for macro query: '{query}'")

    urls = []
    try:
        results = DDGS().text(query, max_results=max_results)
        if results:
            urls = [r.get("href") for r in results if r.get("href")]
            print(f"[HTML-RAG DISCOVERY] Found {len(urls)} target candidate URLs.")
    except Exception as e:
        print(f"[HTML-RAG ERROR] Discovery retrieval failure: {str(e)}")

    return {"discovered_urls": urls}


def fetch_and_convert_to_markdown(state: HtmlRagState) -> dict:
    """Downloads HTML, processes via BS4, flattens via Docling into clean Markdown."""
    urls = state["discovered_urls"]
    compiled_markdown_blocks = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for idx, url in enumerate(urls):
        print(f"[HTML-RAG FETCH & CONVERT] ({idx + 1}/{len(urls)}) Processing: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            cleaned_html_str = str(soup)
            temp_html_path = os.path.join(current_script_dir, f"temp_rag_bs4_{os.getpid()}.html")
            with open(temp_html_path, "w", encoding="utf-8") as f:
                f.write(cleaned_html_str)

            # Flatten using Docling converter
            print("[DOCLING CONVERTER] Generating layout-preserved structural markdown...")
            converter = DocumentConverter()
            docling_result = converter.convert(temp_html_path)
            markdown_text = docling_result.document.export_to_markdown()

            os.remove(temp_html_path)

            markdown_block = f"--- START DOCUMENT: {url} ---\n{markdown_text}\n--- END DOCUMENT ---\n"
            compiled_markdown_blocks.append(markdown_block)
        except Exception as e:
            print(f"[HTML-RAG WARNING] Failed to convert URL {url}: {str(e)}")

    aggregated_context = "\n".join(compiled_markdown_blocks)
    print(f"[HTML-RAG COMPILATION] Aggregated markdown context footprint length: {len(aggregated_context)} characters.")
    return {"raw_markdown_corpus": aggregated_context}


def direct_inference_extraction(state: HtmlRagState) -> dict:
    """Feeds aggregated markdown content straight to the LLM structured schema extractor."""
    print("\n[HTML-RAG INFERENCE] Executing single-turn extraction via schema context map...")
    aggregated_markdown = state["raw_markdown_corpus"]

    if not aggregated_markdown.strip():
        print("[HTML-RAG WARNING] Text corpus is empty. Yielding empty list.")
        return {"extracted_table": []}

    prompt_path = os.path.join(current_script_dir, CONFIG["paths"]["prompt_file"])
    with open(prompt_path, "r", encoding="utf-8") as pf:
        system_prompt_skeleton = pf.read()

    try:
        system_prompt = system_prompt_skeleton.format(
            task_query=state["task_query"],
            raw_markdown_content=aggregated_markdown
        )


        llm = ChatOpenAI(
            model=CONFIG["llm"]["model_name"],
            temperature=float(CONFIG["llm"]["temperature"])
        )
        structured_llm = llm.with_structured_output(EntityTable)

        extracted_pydantic_output = structured_llm.invoke(system_prompt)
        serialized_rows = [row.model_dump() for row in extracted_pydantic_output.rows]

        print(f"[HTML-RAG INFERENCE SUCCESS] Extracted {len(serialized_rows)} tabular elements.")
        return {"extracted_table": serialized_rows}
    except Exception as e:
        print(f"[HTML-RAG CRITICAL CRASH] Inference engine structural failure: {str(e)}")
        return {"extracted_table": []}



# C. GRAPH ASSEMBLER
def compile_html_rag_baseline():
    workflow = StateGraph(HtmlRagState)

    workflow.add_node("discover", execute_discovery_search)
    workflow.add_node("fetch_and_convert", fetch_and_convert_to_markdown)
    workflow.add_node("extract", direct_inference_extraction)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "fetch_and_convert")
    workflow.add_edge("fetch_and_convert", "extract")
    workflow.add_edge("extract", END)

    return workflow.compile()
