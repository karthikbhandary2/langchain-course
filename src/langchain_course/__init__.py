import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate theanswer"
    )


llm = ChatOpenRouter(temperature=0.3, model="nvidia/nemotron-3-ultra-550b-a55b:free")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
