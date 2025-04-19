import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def cargar_documentos_desde_carpeta(carpeta):
    documentos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.pdf'):
            ruta_archivo = os.path.join(carpeta, archivo)
            loader = PyMuPDFLoader(ruta_archivo)
            documentos.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    docs = text_splitter.split_documents(documentos)
    return docs

def procesar_documentos(carpeta):
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embed_model,
                         persist_directory="chroma_db_dir",
                         collection_name="stanford_report_data")

    # Verificar si ya existen embeddings en la base de datos
    total_rows = len(vectorstore.get()['ids'])
    if total_rows == 0:
        print("Procesando documentos y generando embeddings...")
        docs = cargar_documentos_desde_carpeta(carpeta)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory="chroma_db_dir",
            collection_name="stanford_report_data"
        )
        print("Embeddings generados y almacenados en Chroma.")
    else:
        print("Los embeddings ya están generados y almacenados en Chroma.")

if __name__ == "__main__":
    carpeta = "src"
    procesar_documentos(carpeta)