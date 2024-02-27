# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:58:11 2024

@author: Mose
"""
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
#from langchain.callbacks import get_openai_callback

st.set_page_config(page_title="Question your PDF")
key = st.sidebar.text_input('OpenAI API Key', type='password')

os.environ["OPENAI_API_KEY"] = key

#value = st.text_input("Some input")

if key:
    st.write("OpenAI Key has been read ...")

def main():
    #st.set_page_config()
    
    
   #key = st.text_input("Please insert your OPENAI KEY")
    
    
    #load_dotenv(key)
    
   
    st.header("Chat with your PDF")
    
    # upload file
    pdf = st.file_uploader("Add your PDF", type="pdf")
    
    # Append pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        txt = ""
        for page in pdf_reader.pages:
            txt = txt + page.extract_text()
            
        #st.write(txt)
        
        #split into chuncks
        text_split = CharacterTextSplitter(
            separator='\n',
            chunk_size= 1200,
            chunk_overlap=150,
            length_function=len
            )
        
        #created chunks
        chunks = text_split.split_text(txt)
        #st.write(chunks)
        
        #Create embeddings
        embeddgs = OpenAIEmbeddings()
        
        #Search Through the Embeddings
        document = FAISS.from_texts(chunks, embeddgs)
        
        #Text Input for the questions 
        userqn = st.text_input("Please ask questions from your PDF")
        
        if userqn:
            doc = document.similarity_search(userqn)
            #st.write(doc)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type = "stuff")
            answer = chain.run(input_documents=doc, question=userqn)
            
            st.write(answer)
            
            #Get how much you are spending on each question
            #with get_openai_callback() as cb:
                #response = chain.run(input_documents=doc, question=userqn)
                #print(cb)
                
        
        
if __name__ == '__main__':
    main()