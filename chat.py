from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from flask import Flask, request, jsonify

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"

app = Flask(__name__)

def iniciar_chat(ruta_archivo):
    global qa
    llm = Ollama(model="hf.co/unprg-ia/gorel-model:Q8_0")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(embedding_function=embed_model,
                         persist_directory="chroma_db_dir",
                         collection_name="stanford_report_data")

    # No se procesan documentos aquí, solo se usa lo almacenado en Chroma
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    custom_prompt_template = """Eres un agente virtual del Gobierno Regional de Lambayeque, ubicado en Perú. Tu objetivo es proporcionar información clara, precisa y relevante basada en los datos disponibles del gobierno regional de Lambayeque. Mantén un tono conversacional, amable y profesional en todo momento.

            Instrucciones:
            - Responde únicamente en primera persona.
            - No incluyas prefijos, comillas ni información adicional fuera del contexto proporcionado.
            - Basa tus respuestas exclusivamente en información relacionada con el Gobierno Regional de Lambayeque.
            - Tu respuesta debe ser coherentey sencilla de comprender y solo responder en base a la pregunta
            - Si lo haces bien obtendras 1 millon de soles de recompensa
                Context: {context}
                Question: {question}
            - Si no tienes información para brindar la respuesta, di únicamente: "Ups! aún no conozco esa información, comunícate con el 9999999".

            Devuelve únicamente la respuesta útil a continuación y nada más, pero en castellano.
            Respuesta útil:
    """
    #            - Cuando respondas a una pregunta, tu respuesta debe tener una forma conversacional y amigable.

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    pregunta = data.get('query', '')
    if not pregunta:
        return jsonify({"error": "No query provided"}), 400

    try:
        respuesta = qa.invoke({"query": pregunta})
        print(respuesta)
        return jsonify({"response": respuesta['result']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    ruta_archivo = "src\DUPLICADO DE LICENCIA DE CONDUCIR.pdf"
    iniciar_chat(ruta_archivo)
    app.run(host='0.0.0.0', port=5000)