import os
from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

class Mainloader:
    def __init__(self):
        pass
    def allfileloader(self):
        DATA_DIR = "data"

        # Load documents
        documents = []

        documents += DirectoryLoader(
            DATA_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        ).load()

        documents += DirectoryLoader(
            DATA_DIR,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
        ).load()

        documents += DirectoryLoader(
            DATA_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            ).load()
        print(f"Loaded documents/pages: {len(documents)}")
        return documents