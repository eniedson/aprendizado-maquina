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
    """
    Classe para implementar o Retrieval-Augmented Generation (RAG), que combina 
    a recuperação semântica e lexical de documentos com geração de texto baseado em 
    um modelo de linguagem, para responder a perguntas de natureza jurídica.

    Parâmetros:
    - model (str): Nome do modelo de linguagem usado para a geração de texto.
    - data (DataFrame): Dados que contêm os documentos a serem indexados.
    - embedding_model (str): O modelo de embeddings a ser utilizado para transformar o texto em vetores.
    - top_k (int): Número de documentos mais relevantes a serem recuperados.
    - embeddings_size (int): Tamanho dos vetores de embedding.
    - temperature (float): Parâmetro de controle da aleatoriedade na geração do texto.
    """

    def __init__(self, model, data, embedding_model="sentence-transformers/all-MiniLM-L6-v2", top_k=3, embeddings_size = 384, temperature=0.5):
        """
        Inicializa a classe Rag com os parâmetros necessários para recuperação e geração de texto.

        Parâmetros:
        - model (str): Nome do modelo de linguagem usado para a geração de texto.
        - data (DataFrame): Dados contendo os documentos a serem indexados.
        - embedding_model (str): Nome do modelo para embeddings.
        - top_k (int): Número de documentos mais relevantes a serem recuperados.
        - embeddings_size (int): Tamanho dos vetores de embedding.
        - temperature (float): Parâmetro de controle da aleatoriedade na geração do texto.
        """

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
        """
        Concatena o relatório e o voto em uma única string formatada.

        Parâmetros:
        - relatorio (str): O relatório a ser incorporado no texto.
        - voto (str): O voto a ser incorporado no texto.

        Retorno:
        - str: Texto formatado com o relatório e o voto.
        """
        return "Relatório:\n\n" + relatorio + '\n\Voto:\n\n' + voto

    def ingest(self, reindex=True):
        """
        Ingere os documentos no Elasticsearch, indexando-os com embeddings e metadados.

        Parâmetros:
        - reindex (bool): Define se os documentos devem ser reindexados (apaga o índice anterior).

        Retorno:
        - float: O tempo que levou para indexar os documentos, em segundos.
        """
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
        """
        Realiza uma busca no Elasticsearch usando tanto busca semântica quanto lexical.

        Parâmetros:
        - query (str): A consulta de busca.
        - id (int): O ID do documento que não deve ser retornado.

        Retorno:
        - list: Lista de documentos mais relevantes, com base na combinação de busca semântica e lexical.
        """

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
        """
        Aplica o método RRF (Ranked Retrieval Fusion) para combinar os resultados semânticos e lexicais 
        de forma a gerar uma classificação final.

        Parâmetros:
        - docs_semantic (list): Documentos recuperados semânticamente.
        - docs_lexical (list): Documentos recuperados lexicalmente.
        - k (int): Número de documentos a serem retornados.

        Retorno:
        - list: Lista de documentos classificados com base na fusão dos scores semânticos e lexicais.
        """
        
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
        """
        Faz uma consulta ao modelo de linguagem, utilizando o relatório e voto fornecidos 
        juntamente com documentos relevantes recuperados.

        Parâmetros:
        - relatorio (str): O relatório do caso.
        - voto (str): O voto do caso.
        - id (int): O ID do documento que não deve ser retornado.

        Retorno:
        - dict: Resposta gerada pelo modelo com os documentos relevantes usados para a resposta.
        """

        query = self.get_text(relatorio, voto)
        combined_docs = self.search(query, id)
        context = ""
        for i, doc in enumerate(combined_docs):
            context += f"Exemplo {i+1}: \n\n{doc['text']}\n\nAcordão:\n\n{doc['acordao']}"

        prompt_input = self.prompt.format(contexto=context, voto=voto, relatorio=relatorio)
        answer = self.chain.invoke(prompt_input)

        return {"answer": answer, "documents": combined_docs}

    def clear(self):
        """
        Limpa o índice no Elasticsearch.
        """
        self.es.indices.delete(index=self.es_index, ignore=[400, 404])
