from langchain.tools import BaseTool
from langchain.utilities import MetaphorSearchAPIWrapper


class MetaphorSearch(BaseTool):
    name = "metaphor search"
    description = "useful for researching a certain topic"
    return_direct = False

    def _run(self, search_query: str) -> str:
        search = MetaphorSearchAPIWrapper(
            metaphor_api_key=self.metadata["metaphorApiKey"]
        )
        return search.results(search_query, 10, use_autoprompt=True)

    async def _arun(self, search_query: str) -> str:
        search = MetaphorSearchAPIWrapper(
            metaphor_api_key=self.metadata["metaphorApiKey"]
        )
        return await search.results_async(search_query, 10, use_autoprompt=True)
