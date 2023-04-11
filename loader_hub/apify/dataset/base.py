"""Apify dataset reader"""
from typing import Any, Callable, Dict, List

from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document


class ApifyDataset(BaseReader):
    """Apify Dataset reader. Reads data from Apify dataset.

    Args:
        apify_token (str): Apify token.
    """

    def __init__(self, apify_token: str) -> None:
        """Initialize Apify dataset reader."""
        from apify_client import ApifyClient

        self.apify_client = ApifyClient(apify_token)

    def load_data(
        self, dataset_id: str, dataset_mapping_function: Callable[[Dict], Document]
    ) -> List[Document]:
        """Load data from the Apify dataset.
        Args:
            dataset_id (str): Dataset ID.
            dataset_mapping_function (Callable[[Dict], Document]): Function to map dataset items to Document.
        Returns:
            List[Document]: List of documents.
        """
        items_list = self.apify_client.dataset(dataset_id).list_items()

        document_list = []
        for item in items_list.items:
            document = dataset_mapping_function(item)
            if not isinstance(document, Document):
                raise ValueError("Dataset_mapping_function must return a Document")
            document_list.append(document)

        return document_list
