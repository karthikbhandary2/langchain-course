import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"search for {query}")
    return tavily.search(query=query)


llm = ChatOpenRouter(temperature=0.3, model="nvidia/nemotron-3-ultra-550b-a55b:free")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for a Golang Developer in South India or remote on linkedin other job portals and list their details."
                )
            ]
        }
    )
    print(result)
