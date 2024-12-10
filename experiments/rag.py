from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from datetime import datetime
import tqdm


class Rag:
    def __init__(self, model, data, embedding_model="sentence-transformers/all-MiniLM-L6-v2", top_k=3, embeddings_size = 384, temperature=0.5):
        self.model = ChatOllama(model=model, temperature=temperature) if model else None
        self.data = data
        self.top_k = top_k
        self.embedding_size = embeddings_size
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)

        self.es = Elasticsearch("http://localhost:9200", http_auth=("elastic", "teste123"), request_timeout=999999)
        self.es_index = "documents"

        self.chain = (self.model | StrOutputParser()) if model else None
        self.prompt = PromptTemplate.from_template(
            """
            Você é um assistente jurídico especializado na redação de acórdãos do Tribunal de Contas do Estado do Acre. A partir das informações fornecidas, você deve elaborar um acórdão completo que siga uma estrutura clara, coesa e que inclua os seguintes elementos em um texto corrido:\n\n

            <context>
            {contexto}
            </context>
            
            Tomando como base o contexto fornecido e seguindo seu template e padrão. Com base no relatório e no voto abaixo, redija o texto do acordão:

            Relatório:
            {relatorio}

            Voto:
            {voto}

            Acordão:
            """
        )
    
    def get_text(self, relatorio, voto):
        return "Relatório:\n\n" + relatorio + '\n\Voto:\n\n' + voto

    def ingest(self, reindex=True):
        if reindex:
            tic = datetime.now()
            print('Total Documents: ' + str(len(self.data)))


            if self.es.indices.exists(index=self.es_index):
                self.es.indices.delete(index=self.es_index)

            mappings = {
                "mappings": {
                    "properties": {
                        "id": {
                            "type": "long"
                        },
                        "acordao": {
                            "type": "text"
                        },
                        "text": {
                            "type": "text"
                        },
                        'embedding': {
                            "type": "dense_vector",
                            "dims": self.embedding_size,
                            "index": True,
                            "similarity": "cosine",
                        }
                    }
                }
            }
                
            self.es.indices.create(index=self.es_index, body=mappings)

            actions = []
            with tqdm.tqdm(total= len(self.data), smoothing=0.2) as pbar:
                for i, row in self.data.iterrows():
                    text = self.get_text(row['relatorio'], row['voto'])
                    embedding_vector = self.embedding.embed_documents([text])[0]
                    action = {
                        "_index": self.es_index,
                        "_id": row['id'],
                        "_source": {
                            "id": row['id'],
                            "acordao": row['acordao'],
                            "text": text,
                            'embedding': embedding_vector
                        }
                    }

                    actions.append(action)
                    pbar.update(1)

            bulk(self.es, actions)
            print('Elasticsearch index ready!')
            return (datetime.now() - tic).total_seconds()

    def search(self, query, id):
        docs_semantic = None
        query_embedding = self.embedding.embed_query(query)
        response_semantic = self.es.search(
            index=self.es_index,
            body={
                "size": self.top_k,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": self.top_k,
                    "num_candidates": 100
                },
                "query": {
                    "bool": {
                        "must_not": [
                            {
                                "term": {
                                    "id": id
                                }
                            }
                        ]
                    }
                }
            }
        )
        docs_semantic = [
            hit["_source"]
            for hit in response_semantic["hits"]["hits"]
        ]

        response_lexical = self.es.search(
            index=self.es_index,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "text": query
                                }
                            }
                        ],
                        "must_not": [
                            {
                                "term": {
                                    "id": id
                                }
                            }
                        ]
                    }
                },
                "size": self.top_k
            }
        )
        docs_lexical = [
            hit["_source"]
            for hit in response_lexical["hits"]["hits"]
        ]

        return self.rrf(docs_semantic, docs_lexical)

    def rrf(self, docs_semantic, docs_lexical, k=10):
        scores = {}
        docs = {}
        for rank, doc in enumerate(docs_semantic):
            scores[doc['id']] = scores.get(doc['id'], 0) + 1 / (k + rank)
            docs[doc['id']] = doc
        for rank, doc in enumerate(docs_lexical):
            scores[doc['id']] = scores.get(doc['id'], 0) + 1 / (k + rank)
            docs[doc['id']] = doc

        combined_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result_docs = list(map(lambda id: docs[id], combined_docs))
        return result_docs[:self.top_k]

    def ask(self, relatorio: str, voto: str, id:int):
        query = self.get_text(relatorio, voto)
        combined_docs = self.search(query, id)
        context = ""
        for i, doc in enumerate(combined_docs):
            context += f"Exemplo {i+1}: \n\n{doc['text']}\n\nAcordão:\n\n{doc['acordao']}"

        prompt_input = self.prompt.format(contexto=context, voto=voto, relatorio=relatorio)
        answer = self.chain.invoke(prompt_input)

        return {"answer": answer, "documents": combined_docs}

    def clear(self):
        self.es.indices.delete(index=self.es_index, ignore=[400, 404])
