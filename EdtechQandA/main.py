import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain
st.title("FAQs about the courses")
btn = st.button("create a knowledgebase") #this is used when a data scientist want to craete a new vector database for the new questions added


if btn:
    pass

question = st.text_input("Enter your question :")

if question:
    chain = get_qa_chain()
    response =chain(question)

    st.header("Answer:")
    st.write(response["result"])   # this is how we put answer below the question
