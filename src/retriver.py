
class mainretriever:
    def __init__(self):
        pass
    def Topkretriever(self,question,vector_db,TOP_K):
        # Retriever
        retriever = vector_db.as_retriever(
            search_kwargs={"k": TOP_K}
        )
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )
        return context
        