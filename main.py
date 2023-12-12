from langchain.llms import GooglePalm
import os
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()


llm = GooglePalm(google_api_key=os.environ['GOOGLE_API_KEY'], temperature=0.7)

instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large")
vector_file_path = "faiss_index"


def create_db():
    loader = CSVLoader(file_path='PlantFAQ2.csv', source_column="Query")
    data = loader.load()
    vectordb = FAISS.from_documents(
        documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vector_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vector_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.65)

    prompt_template = """You are a Plant Researcher having extensive knowledge about the Ayurveda plants.For 
                        the given question provide an answer based on the context only.For the answer try to provide as much text 
                        as possible from "answer" section in the source document context without making many changes.

                        If answer is not known kindly print "I'm unable to provide an answer on that topic as it's beyond my current scope. 
                        Consulting a subject matter expert would offer the most accurate insights." 

                        After providing the answer always add "Please note that the information provided here is based on general knowledge about these plants and their traditional uses. For specific details, it's advisable to seek guidance from an expert or doctor well-versed in this area." except if answer is not known.

                        CONTEXT: {context}

                        QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}

    memory = ConversationBufferWindowMemory(k=10)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        input_key="query",
                                        retriever=retriever,
                                        chain_type_kwargs=chain_type_kwargs,
                                        memory=memory)
    # print(memory.load_memory_variables({}))
    return chain


# global_chain = None
# chain = get_qa_chain()
# global_chain = chain


if __name__ == "__main__":
    # create_db()
    chain = get_qa_chain()
    response = chain("where are the uses of tulsi")
    hisres = response["history"]
    print(response)
    response = chain("where is it found")
    print(response)
    hisres = response["history"]
