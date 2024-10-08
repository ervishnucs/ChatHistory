# Conversational RAG (Retrieval-Augmented Generation) Chain with LangChain

This project demonstrates the use of LangChain to build a conversational retrieval-augmented generation (RAG) system that can retrieve relevant information from documents and handle ongoing conversations with context awareness. The system processes PDF documents, splits the text, stores embeddings in a vector database, and allows question-answering by retrieving information from the document, while keeping track of conversation history.

## Features
- **Document Processing**: PDF documents are loaded and split into chunks for easier processing.
- **Vector Search**: Chunks of document data are embedded and stored in a vector store (Chroma) for fast retrieval based on user queries.
- **Context-Aware QA**: The system retrieves relevant context from documents and provides concise answers based on it.
- **Chat History Management**: Keeps track of the chat conversation to maintain the context for subsequent questions.

## Installation

1. **Clone the repository**:

    ```bash
    git clone <repo_url>
    cd <repo_directory>
    ```

2. **Install the required dependencies**:
    Install the necessary Python libraries using `pip`:

    ```bash
    pip install langchain langchain_chroma langchain_core langchain_text_splitters
    ```

3. **Set up the environment**:
    Ensure you have the PDF files ready in the correct paths, and update the paths in the script accordingly.

## Usage

1. **Load the PDF documents**:
   The `PyPDFLoader` is used to load the PDF files into the system:

    ```python
    loader = PyPDFLoader(
        r"path_to_first_pdf",
        r"path_to_second_pdf",
        r"path_to_third_pdf"
    )
    docs = loader.load()
    ```

2. **Split Documents**:
   Use the `RecursiveCharacterTextSplitter` to split the documents into smaller chunks:

    ```python
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    ```

3. **Embedding and Retrieval**:
   Create a vector store for document embeddings using Chroma and set up the retriever for search:

    ```python
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
    retriever = vectorstore.as_retriever()
    ```

4. **History-Aware Retrieval**:
   Create a history-aware retriever to allow context-based question reformulation:

    ```python
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    ```

5. **Question Answering Chain**:
   Set up the question-answering chain using the `create_stuff_documents_chain` function:

    ```python
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    ```

6. **Conversational RAG Chain**:
   Build the conversational RAG chain and invoke it with chat history and questions:

    ```python
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    ```

7. **Run the Conversation**:
   Invoke the chain with the user’s question and chat history:

    ```python
    response = conversational_rag_chain.invoke(
        {"input": "What is the last question I asked?"},
        config={"configurable": {"session_id": "abc123"}},
    )["answer"]
    ```

8. **Access Chat History**:
   The chat history can be printed as follows:

    ```python
    for message in store["abc123"].messages:
        print(f"{prefix}: {message.content}\n")
    ```

## File Structure

- `main.py`: The main script that loads documents, sets up the retrieval and question-answering system, and handles conversation history.
- `policies/`: Folder containing PDF documents that will be processed by the system.

## Requirements

- Python 3.12 or higher
- LangChain
- Chroma
- PyPDFLoader
- Hugging Face embeddings model

