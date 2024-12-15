import glob 
import chromadb
import json 
from langchain_ollama import OllamaEmbeddings

class ChromaVectorDatabase:

    CHROMA_PATH='./tmp/chromadb'

    def __init__(self, collection_name:str, embedders:OllamaEmbeddings):
        self.client = chromadb.PersistentClient(path=ChromaVectorDatabase.CHROMA_PATH)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedders = embedders
    
    def load_data(self,) -> list[str]:

        folder_path = './data/*.jsonl'
        jsonl_files = glob.glob(folder_path)
        all_data = []
        for jfile in jsonl_files:
            with open(jfile, 'r') as f:
                for line in f:
                    # Parse each line as a JSON object
                    data = json.loads(line)
                    all_data += data['main_information']
                    all_data += data['additional_information']

        # remove null values 
        return [text for text in all_data if text] 

    def add_to_vector_db(self, texts:list[str]) -> None:
        for i, d in enumerate(texts):
            embeddings = self.embedders.embed_query(d)
            self.collection.add(
                ids=[str(i)],
                embeddings=[embeddings],
                documents=[d]
            )

    def delete_vector_db(self) -> None:
        self.client.delete_collection(name=self.collection_name)

    def generate_context(self, query:str, n_results:int=3) -> str:

        results = self.collection.query(
            query_embeddings=self.embedders.embed_query(query),
            n_results=n_results,
        )

        return " \n".join(str(item) for item in results['documents'][0])

