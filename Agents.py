

# --------------------------- IMPORTS --------------------------- #
import os
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

from dust import agent


# --------------------------- GROQ SETTINGS --------------------------- #
# ✅ Environment variable only (NO strings, NO headers)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ------------------------- LLM: ChatGroq ------------------------------ #
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY
)


# -------------------- AGENT 5: SOLUTION EXPLAINER -------------------- #
solution_explainer_prompt = PromptTemplate.from_template(
    """
You are a highly skilled math teacher.

### TASK:
Explain the following math problem in clear, simple, and step-by-step format.

### INPUT:
{combined_input}

### INSTRUCTIONS:
- Ignore any visual instructions or drawing hints.
- Only focus on solving the problem description and step by step solution logically and mathematically.
- Each step should be broken down with reasoning and intermediate calculations.
- Format output like:
  Step 1: ...
  Step 2: ...
  ...
  Final Answer: ...

-also provide analogy explain the mathematical sum in an simple way that even a child can understand
"""
)

solution_explainer_chain = solution_explainer_prompt | llm


# --------------------------- FUNCTIONS --------------------------- #

def process_question(question: str) -> str:
    """
    Uses ChatGroq (LangChain) instead of raw HTTP requests
    """
    prompt = f"""
    you should do this do not even skip it
    You are an AI tutor. Format your response in three simple sections with clear and accurate content.

    Problem Description:
    Summarize the given question. Mention all given data and what needs to be solved.
    give as an instruction for the another chatgroq model

    Solution Steps:
    Provide a step-by-step simple and effective solution for the problem.
    give as an instruction for the another chatgroq model

    Visual Hints(must without this do not proceed) :
    provide visual hints that is the object that can be added and the labels for it
    the video should be visually powerful and it entirely depends on your instruction
    since real world objects cannot be created in manim give hints like draw rectangle for train with circle as its wheels 
    it can also have spg objects if possible 
    give as an instruction for the another chatgroq model
    look at what hints can be provided 



    Now format the response for the question below:

    Question: {question} 
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def solution_to_html(solution: str) -> str:
    prompt = f"""
    you should do the given solution to html code 
    just want <p> ,<li>,<ul> like this you wnt to give as a result
    
    IMPORTENT : give only html code no other additional preamble i dont want that like "Here is the HTML code without preamble:" this also
    {solution}
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def AgentManager(question: str) -> str:
    cleaned_response = process_question(question)

    solution_explanation = solution_explainer_chain.invoke(
        {"combined_input": cleaned_response}
    )

    formatted_solution = solution_explanation.content.strip()
    agent(cleaned_response)

    return solution_to_html(formatted_solution)
