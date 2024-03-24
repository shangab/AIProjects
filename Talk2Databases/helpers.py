import os
from dotenv import load_dotenv
from langchain.utilities.sql_database import SQLDatabase

# llm imports

from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


class SmartDatabase:
    def __init__(self, db_uri: str):
        self.db = SQLDatabase.from_uri(database_uri=db_uri)

    def get_shcema(self):
        return self.db.get_table_info()

    def run_query(self, query: str):
        return self.db.run(query)


class LLMTool:
    def __init__(self, db_uri: str):
        self.llm = OpenAI()
        self.db = SmartDatabase(db_uri=db_uri)

        # 1. Query chain
        self.query_template = """
        Given the belwo question and the database schema, please generate a SQL query that answers the question.
        Question: {question}
        Database Schema: {schema}
        SQL Query is:"""

        self.query_prompt = ChatPromptTemplate.from_template(
            self.query_template)
        self.query_chain = LLMChain(llm=self.llm, prompt=self.query_prompt)

        # 1. Answer chain
        self.answer_template = """
        Given the belwo question and the database results, please rewrite the results in the same 
        language the question was written with:\n
        
        Question: {question}
        Database Results: {results}
        
        Answer is:"""
        self.answer_prompt = ChatPromptTemplate.from_template(
            self.answer_template)
        self.answer_chain = LLMChain(llm=self.llm, prompt=self.answer_prompt)

    def get_query(self, question: str):
        schema = self.db.get_shcema()
        return self.query_chain.run(question=question, schema=schema)

    def get_answer(self, question: str, results: str):
        return self.answer_chain.run(question=question, results=results)

    def runall(self, question: str):
        query = self.get_query(question=question)
        results = self.db.run_query(query)
        answer = self.get_answer(question=question, results=results)
        return query, results, answer


if __name__ == '__main__':
    db_uri = "mysql+mysqlconnector://root:root@localhost:3306/MSD"
    llm_tool = LLMTool(db_uri=db_uri)
    question = "كم عدد الموظفين لدينا الآن؟"
    query, results, answer = llm_tool.runall(question=question)
    print(answer)
# "Show me all customers who made orders in 2003."
# "What are the top-selling products in terms of revenue?"
# "List all product lines available in the database."
# "Retrieve details of all orders placed by customer: Atelier graphique”
# “How many orders were placed in the year 2004?"
# "Find all employees who report to: Anthony Bow."
# "Show me the total revenue generated from orders placed in the last quarter of 2003."
# "List all customers who haven't made any payments yet."
# "Retrieve the details of all sales offices located in ‘NYC’."
# "What is the average order value for each productline?"
