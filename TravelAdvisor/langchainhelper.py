import os
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

openai_llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo-instruct",
                    api_key=OPENAI_API_KEY)


# Country prompting

country_template = PromptTemplate(
    template="If I want a noce holiday described as: {details}. Please give me only one country name, and no other details. ",
    input_variables=["details"]
)
country_chain = LLMChain(llm=openai_llm, prompt=country_template)

# Preps prompting
preps_template = PromptTemplate(
    template="If I want  to go to {country}, in five top points tell me how I should be prepared?.",
    input_variables=["country"]
)
preps_chain = LLMChain(llm=openai_llm, prompt=preps_template)


# attractions prompting
attractions_template = PromptTemplate(
    template="Tell me the top 6 attractions in {country} to visit.",
    input_variables=["country"]
)
attractions_chain = LLMChain(llm=openai_llm, prompt=attractions_template)


# avoid  prompting
avoid_template = PromptTemplate(
    template="Tell me the top 6 things to avoid when visiting {country}.",
    input_variables=["country"]
)
avoid_chain = LLMChain(llm=openai_llm, prompt=avoid_template)


def get_advise(details):
    country = country_chain.run(details)
    preps = preps_chain.run(country)
    attractions = attractions_chain.run(country)
    avoid = avoid_chain.run(country)

    return country, preps, attractions, avoid
