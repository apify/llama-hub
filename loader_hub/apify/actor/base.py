"""Apify Actor reader"""
from typing import Any, Callable, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class ApifyActor(BaseReader):
    """Apify Actor reader. Runs and reads data from Apify Actor run.

    Args:
        apify_token (str): Apify token.
    """

    def __init__(self, apify_token: str) -> None:
        """Initialize Apify Actor reader."""
        from apify_client import ApifyClient

        self.apify_client = ApifyClient(apify_token)

    def load_data(
        self,
        actor_id: str,
        run_input: Dict,
        dataset_mapping_function: Callable[[Dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> List[Document]:
        """Run an Actor on the Apify platform and wait for it to finish and return it's data.
        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single dictionary (an Apify dataset item) and converts it to an instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run. It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run, in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            List[Document]: List of documents.
        """
        actor_call = apify_client.actor(actor_id).call(
            run_input=run_input,
            build=build,
            memory_mbytes=memory_mbytes,
            timeout_secs=timeout_secs,
        )
        items_list = self.apify_client.dataset(
            actor_call.get("dataset_id")
        ).list_items()

        document_list = []
        for item in items_list.items:
            document = dataset_mapping_function(item)
            if not isinstance(document, Document):
                raise ValueError("Dataset_mapping_function must return a Document")
            document_list.append(document)

        return document_list
