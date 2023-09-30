from langchain.llms.replicate import Replicate as ReplicateModel
from langchain.tools import BaseTool


class Replicate(BaseTool):
    name = "Replicate"
    description = "useful for querying a Replicate model."
    return_direct = False

    def _run(self, prompt: str) -> str:
        model = self.metadata["model"]
        api_token = self.metadata["apiKey"]
        input = self.metadata["arguments"]
        model = ReplicateModel(
            model=model, input=input, api_token=api_token, replicate_api_token=api_token
        )
        return model.predict(prompt)

    async def _arun(self, prompt: str) -> str:
        model = self.metadata["model"]
        api_token = self.metadata["apiKey"]
        input = self.metadata["arguments"]
        model = ReplicateModel(
            model=model, input=input, api_token=api_token, replicate_api_token=api_token
        )
        return await model.apredict(prompt)
