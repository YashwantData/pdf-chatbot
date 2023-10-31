import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os

# storing history in session states
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with  st.chat_message("assistant"):
            st.markdown(message["content"])

# Sidebar contents
with st.sidebar:
    st.title('PDF BASED LLM CHATBOT🤗')
    key = st.text_input("Add your API Key")
    print(key)
    # st.header("Chat with pdf")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    os.environ["OPENAI_API_KEY"] = key
    st.markdown('''
    ## About APP:

    The app's primary resource is utilised to create::

    - [streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/yashwant-rai-2157aa28b)

    ''')

    st.write('💡All about pdf based chatbot, created by Yashwant Rai')


def main(pdf):
    # st.write(pdf)
    if pdf is not None:
        # pdf reader
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        # st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            # with open(f"{store_name}.pkl", "wb") as f:
            # pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.chat_input(placeholder="Ask questions about your PDF file:")
        # st.write(query)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if query:
            # pdf = st.file_uploader("Upload your PDF", type='pdf', label_visibility=dis)
            chat_history = []
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in English language.' If you do not know the answer reply with 'I am sorry'.
                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone question:
                        Remember to greet the user with hi welcome to pdf chatbot how can i help you? if user asks hi or hello """

            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
            # docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(temperature=0)
            # chain = load_qa_chain(llm=llm, chain_type="stuff",memory=memory)
            # with get_openai_callback() as cb:
            # response = chain.run(input_documents=docs, question=query)
            # print(cb)
            # print(response)
            # st.write(response)
            qa = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=memory
            )
            response = qa({"question": query, "chat_history": chat_history})
            # st.write(response["answer"])
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            chat_history.append((query, response))


if __name__ == '__main__':
    main(pdf)
