# web_search.py
# M1 - Anya

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import os
from dotenv import load_dotenv

load_dotenv()  # automatically reads .env in current directory

def get_serpapi_key():
    key = os.getenv("SERPAPI_KEY")
    if not key:
        raise ValueError("SERPAPI_KEY not found in environment")
    return key

SERP_API_URL = "https://serpapi.com/search"


def search_planet_docs(query: str, api_key: str = None) -> dict:
    """
    
    """

    if api_key is None:
        api_key = get_serpapi_key()

    # Force domain restriction to Planet docs
    refined_query = f"site:docs.planet.com {query}"

    params = {
        "engine": "google",
        "q": refined_query,
        "api_key": api_key,
        "num": 3
    }

    response = requests.get(SERP_API_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"SerpAPI error: {response.text}")

    data = response.json()
    organic = data.get("organic_results", [])

    if not organic:
        return {
            "content": "",
            "source_url": "",
            "section": ""
        }

    top_result = organic[0]
    url = top_result.get("link")

    page_html = fetch_page_content(url)
    content_md = html_to_markdown(page_html)

    section = extract_relevant_section(content_md, query)

    return {
        "content": section,
        "source_url": url,
        "section": top_result.get("title", "")
    }


def fetch_page_content(url: str) -> str:
    """
    
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Workflow-Agent)"
    }

    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {url}")

    return response.text


def html_to_markdown(html: str) -> str:
    """
    
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    clean_html = str(soup)
    return md(clean_html)


# def extract_relevant_section(content: str, query: str) -> str:
#     """
#     Naive but effective keyword-based section extraction.

#     Since we are NOT using embeddings (by design), we:
#     - split into chunks by headers
#     - score chunks by keyword overlap
#     """

#     query_terms = set(query.lower().split())

#     sections = content.split("\n#")  # rough header split

#     best_section = ""
#     best_score = 0

#     for section in sections:
#         lower = section.lower()

#         score = sum(1 for term in query_terms if term in lower)

#         if score > best_score:
#             best_score = score
#             best_section = section

#     return best_section.strip()
def extract_relevant_section(content: str, query: str) -> str:
    """
    
    """
    query_terms = set(query.lower().split())

    sections = content.split("\n## ")

    best_section = ""
    best_score = 0

    for section in sections:
        lower = section.lower()

        # keyword match
        keyword_score = sum(1 for term in query_terms if term in lower)

        # bonus: reward API-related keywords
        api_boost = 0
        if "api" in lower or "auth" in lower or "token" in lower:
            api_boost = 2

        score = keyword_score + api_boost

        if score > best_score:
            best_score = score
            best_section = section

    return best_section.strip()
