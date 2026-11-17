import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")

    information = """
    Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 26 May 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the Member of Parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is India's third-longest-serving prime minister, and the longest-serving prime minister outside the Indian National Congress.[b]
    """

    summary_template = """
    given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # temperature deals with how deterministic the output should be. 0 means deterministic, 1 means it can be creative/out of box.
    llm = ChatOpenRouter(temperature=0, model="nvidia/nemotron-3-ultra-550b-a55b:free")

    """
    chain by composing the template and the llm. | uses the output of the left component i.e summary_prompt_template as the input to the right component i.e llm
    """
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
