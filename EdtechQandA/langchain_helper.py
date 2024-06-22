#to import a csv file we need to import CSVLoader from the langchain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
#importing FAISS as we decided to use it as our vectordatabase
import os
from dotenv import load_dotenv
load_dotenv()

import langchain_google_genai
from langchain_google_genai import GoogleGenerativeAI
llm = GoogleGenerativeAI(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.7,model="gemini-pro")


inst_embeddings = HuggingFaceEmbeddings() # we are keeping this outside of the function as we will be using this for other usescases as well
vectordb_file_path = "faiss_index"    #later on we will be using this file path/DB path, which creates vectorDB and save it to a disk
#the goal of this function will be, create a database and serialize it to disk
#so that DB is stored like a file system
def  create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data ,embedding=inst_embeddings)
    vectordb.save_local(vectordb_file_path)  #when a vectordb is created we can save to the file,which will save it to a local file
    #that file will be faiss_index which will actually create a directory faiss_index

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, inst_embeddings, allow_dangerous_deserialization=True) #load the vector database from the local folder
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """ Given the following context and a question,generate an answer based on the this context only.
    In the answer try to provide as much text as possible from "response" section in the source docuemnt context without making much changes.
    If the answer is not found in the context, kindly state "I don't know you can reach out to an agent." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]

    )
    chain_type_kwargs = {"prompt": PROMPT}

    # we will use this RetrievalQA class of LangChain
    # to pull the similar looking embedding vectors to form a prompt

    from langchain.chains import RetrievalQA

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        # can be changed to false if you dont want to see the source documents from where its referring from
                                        chain_type_kwargs=chain_type_kwargs)
    return chain
if __name__ == "__main__":
    chain = get_qa_chain()

    print(chain.invoke("do you provide internship? do you have EMI option?"))