# flake8: noqa
import requests
import pandas as pd

from io import StringIO
from decouple import config
from tempfile import NamedTemporaryFile
from langchain.tools import BaseTool
from llama import Context, LLMEngine, Type
from app.vectorstores.pinecone import PineconeVectorStore
from app.datasource.loader import DataLoader
from prisma.models import Datasource

from langchain.agents.agent_types import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models.openai import ChatOpenAI


class DatasourceFinetuneTool(BaseTool):
    name = "datasource"
    description = "useful for when you need to answer questions"
    return_direct = False

    def _run(
        self,
        question: str,
    ) -> str:
        """Use the tool."""

        class Question(Type):
            question: str = Context("A question")

        class Answer(Type):
            answer: str = Context("An answer to the question")

        llm = LLMEngine(
            id=self.metadata["agent_id"],
            config={"production.key": config("LAMINI_API_KEY")},
            model_name="chat/gpt-3.5-turbo",
        )
        input = Question(question=question)
        output = llm(input=input, output_type=Answer)
        return output.answer

    async def _arun(
        self,
        question: str,
    ) -> str:
        """Use the tool asynchronously."""

        class Question(Type):
            question: str = Context("A question")

        class Answer(Type):
            answer: str = Context("An answer to the question")

        llm = LLMEngine(
            id=self.metadata["agent_id"],
            config={"production.key": config("LAMINI_API_KEY")},
            model_name="chat/gpt-3.5-turbo",
        )
        input = Question(question=question)
        output = llm(input=input, output_type=Answer)
        return output.answer


class DatasourceTool(BaseTool):
    name = "datasource"
    description = "useful for when you need to answer questions"
    return_direct = False

    def _run(
        self,
        question: str,
    ) -> str:
        """Use the tool."""
        pinecone = PineconeVectorStore()
        return pinecone.query_documents(
            prompt=question,
            datasource_id=self.metadata["datasource_id"],
            query_type=self.metadata["query_type"],
            top_k=3,
        )

    async def _arun(
        self,
        question: str,
    ) -> str:
        """Use the tool asynchronously."""
        pinecone = PineconeVectorStore()
        return pinecone.query_documents(
            prompt=question,
            datasource_id=self.metadata["datasource_id"],
            query_type=self.metadata["query_type"],
            top_k=3,
        )


class StructuredDatasourceTool(BaseTool):
    name = "structured datasource"
    description = "useful for when need answer questions"
    return_direct = False

    def _load_xlsx_data(self, url):
        response = requests.get(url)
        with NamedTemporaryFile(suffix=".xlsx", delete=True) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            df = pd.read_excel(temp_file.name, engine="openpyxl")
        return df

    def _load_csv_data(self, url):
        response = requests.get(url)
        file_content = StringIO(response.text)
        return pd.read_csv(file_content)

    def _run(
        self,
        question: str,
    ) -> str:
        """Use the tool."""
        datasource: Datasource = self.metadata["datasource"]
        if datasource.type == "CSV":
            df = self._load_csv_data(datasource.url)
        elif datasource.type == "XLSX":
            df = self._load_xlsx_data(datasource.url)
        else:
            data = DataLoader(datasource=datasource).load()
            df = pd.DataFrame(data)
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0,
                model="gpt-4-0613",
                openai_api_key=config("OPENAI_API_KEY"),
            ),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return agent.run(question)

    async def _arun(
        self,
        question: str,
    ) -> str:
        """Use the tool asynchronously."""
        datasource: Datasource = self.metadata["datasource"]
        if datasource.type == "CSV":
            df = self._load_csv_data(datasource.url)
        elif datasource.type == "XLSX":
            df = self._load_xlsx_data(datasource.url)
        else:
            data = DataLoader(datasource=datasource).load()
            df = pd.DataFrame(data)
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0,
                model="gpt-4-0613",
                openai_api_key=config("OPENAI_API_KEY"),
            ),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return await agent.arun(question)
