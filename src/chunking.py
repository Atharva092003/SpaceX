from langchain_text_splitters import RecursiveCharacterTextSplitter



class Chunker:
    def __init__(self):
        pass
    def recursiveoverlap(self,documents):
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = text_splitter.split_documents(documents)
        print("Chunking Created..")
        return chunks