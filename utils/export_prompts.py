from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv(".env-langsmith")

def export_prompts():
        
    template_concise = """You are an evaluator tasked with assessing if an article match a set of criteria based on its Summary.

    [begin Summary]
    {title}
    {summary}
    [end Summary]

    The reader is a seasoned Machine Learning Engineer and Researcher.
    Score the potential impact of the news article on their job from 1 to 3.
    News that significantly affects the job of the reader includes critical new research findings, product releases, or major industry developments. 

    Example response:
    {{"relevant": <score from 1 to 3>, "comment": "<your explanation here>"}}

    Response:"""


    template_verbose = """Keep these instructions open and refer to them as often as you need during the assessment.
    You are tasked with assessing the impact of the news article on a seasoned Machine Learning Engineer and Researcher.
    News that significantly affects the job of the reader includes critical new research findings, product releases, or major industry developments.
    News that moderately affects the job of the reader includes items that are somewhat relevant but not critically important to the seasoned Machine Learning reader.
    Your response will provide a comment describing your assessment followed by a score from 1 to 3.

    Score of 3 is for information that is highly likely to impact the reader.
    Score of 2 is for information that moderately likely to impact the reader.
    Score of 1 is for information that is unlikely to impact the reader.

    [begin Summary]
    {title}
    {summary}
    [end Summary]

    The response should be JSON formatted and include:
    * a "comment" key with a short explanation about your evaluation.
    * a "relevant" key with a score from 1 to 3

    Example:
    {{"comment": "<your explanation here>", "relevant": <score from 1 to 3>}}

    Response:"""

    client = Client()
    for prompt_name, prompt_template in [("filter_concise", template_concise), ("filter_verbose", template_verbose)]:
        prompt = ChatPromptTemplate.from_template(prompt_template)
        url = client.push_prompt(prompt_name, object=prompt)
        print(f"Prompt {prompt_name} created at {url}")

if __name__ == "__main__":

    export_prompts()