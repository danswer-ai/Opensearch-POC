from dataclasses import dataclass

from opensearchpy import OpenSearch

from utils import EMBEDDING_DIM, vectorize, TextType

@dataclass
class MinimalChunk:
    content: str

@dataclass
class MinimalDoc:
    title: str
    chunks: list[MinimalChunk]


# Creating documents for testing
DOC_1 = MinimalDoc(
    title="Doc 1",
    chunks=[
        MinimalChunk(content="The weather in Florida is hot and humid"),
        MinimalChunk(content="The weather in Alaska is frigid"),
        MinimalChunk(content="The weather in the Sahara is dry")
    ]
)

DOC_2 = MinimalDoc(
    title="Doc 2",
    chunks=[
        MinimalChunk(content="My favorite animal in the world is the dog"),
        MinimalChunk(content="My favorite animal in the world is the cat"),
        MinimalChunk(content="My favorite animal in Florida is the alligator")
    ]
)

DOC_3 = MinimalDoc(
    title="Doc 3",
    chunks=[
        MinimalChunk(content="The best food is French fries"),
        MinimalChunk(content="The best food is pizza"),
        MinimalChunk(content="The best food is sushi")
    ]
)

DOCUMENTS = [DOC_1, DOC_2, DOC_3]

def get_opensearch_client():
    return OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=("admin", "D@nswer_1ndex"),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )


def create_index(client, index_name):
    hnsw_config = {
        "type": "knn_vector",
        "dimension": EMBEDDING_DIM,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "lucene",
            "parameters": {
                "ef_construction": 200,
                "m": 48
            }
        }
    }

    schema = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "chunks": {
                    "type": "nested",
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": hnsw_config
                    }
                },
            }
        }
    }

    # Wiping just to be sure
    client.indices.delete(index_name, ignore=[404])
    print(f"Creating Index {index_name}")
    response = client.indices.create(index_name, body=schema)
    print(response)


def add_normalization_processor(client):
    pipeline_body = {
        "description": "Normalization for keyword and vector scores",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {
                        "technique": "min_max"
                    },
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [
                                0.3,
                                0.7
                            ]
                        }
                    }
                }
            }
        ]
    }
    client.search_pipeline.put(id="normalization_step", body=pipeline_body)


def index_document(client, index_name, document: MinimalDoc):
    doc = {
        "title": document.title,
        "chunks": [
            {
                "content": chunk.content,
                "embedding": vectorize(chunk.content, TextType.PASSAGE)
            } for chunk in document.chunks
        ],
    }

    print(f"Indexing {document.title} document")
    response = client.index(index=index_name, body=doc)
    print(response)


def hybrid_search_v1(client, index_name, query, max_num_results=10):
        query_vector = vectorize(query, TextType.QUERY)

        # We need to use the nested field and also hybrid
        # Either the hybrid is on the outside or the nested is on the outside
        # Neither work correctly but the behavior is different
        # The nested named queries don't really work right as well

        # For this one, the problems are:
        # It only gives back one chunk and there are no inner hits so we have no idea what the individual chunk scores are
        # The scores seem normalized as in the smallest is 0 and largest is 1 (if a doc has both the highest keyword and vector scores or it has the lowest of both)
        # However the middle scores just seem wrong, there's no reasonable way of normalizing that leads to that middle score
        search_body_hybrid_outside = {
            "size": max_num_results,
            "query": {
                "hybrid": {
                    "queries": [
                        # Chunk Keyword Score
                        {
                            "nested": {
                                "path": "chunks",
                                "query": {
                                    "match": {
                                        "chunks.content": {
                                            "query": query,
                                            "_name": "chunk_keyword_score"
                                        }
                                    }
                                },
                                "score_mode": "max",
                                "inner_hits": {
                                    "size": 20
                                }
                            },
                        },
                        # Chunk Vector Score
                        # This way of nesting apparently doesn't give back the inner hits
                        # Gives back documents that are normalized
                        {
                            "nested": {
                                "path": "chunks",
                                "query": {
                                    "knn": {
                                        "chunks.embedding": {
                                            "vector": query_vector,
                                            "k": max_num_results,
                                            "_name": "chunk_vector_score"
                                        },
                                    },
                                },
                                "score_mode": "max",
                                "inner_hits": {
                                    "size": 20
                                }
                            },
                        },
                    ],
                },
            },
        }
            
        response = client.search(
            index=index_name,
            search_pipeline="normalization_step",
            body=search_body_hybrid_outside,
            include_named_queries_score=True
        )

        return response


def hybrid_search_v2(client, index_name, query, max_num_results=10):
    query_vector = vectorize(query, TextType.QUERY)

    # This one is wrong for other reasons, the inner hits are definitely not normalized
    # We tried changing the normalization weighting and it does not change anything
    # Can see inner hits but just one chunk per hit
    search_body_hybrid_inside = {
        "size": max_num_results,
        "query": {
            "nested": {
                "path": "chunks",
                "query": {
                    "hybrid": {
                        "queries": [
                            # Keyword Score
                            {
                                "match": {
                                    "chunks.content": {
                                        "query": query,
                                        "_name": "chunk_keyword_score"
                                    }
                                }
                            },
                            # Chunk Vector Score
                            {
                                "knn": {
                                    "chunks.embedding": {
                                        "vector": query_vector,
                                        "k": max_num_results,
                                        "_name": "chunk_vector_score"
                                    },
                                },
                            },
                        ],
                    },
                },
                "score_mode": "max",
                "inner_hits": {
                    "size": 20
                },
            },
        },
    }

    response = client.search(
        index=index_name,
        search_pipeline="normalization_step",
        body=search_body_hybrid_inside,
        include_named_queries_score=True
    )

    return response


def main():
    client = get_opensearch_client()
    index_name = "danswer-index"

    create_index(client, index_name)

    add_normalization_processor(client)

    for document in DOCUMENTS:
       index_document(client, index_name, document)

    print("Performing hybrid search")
    print(hybrid_search_v1(client, index_name, "Florida"))
    print(hybrid_search_v2(client, index_name, "Florida"))


if __name__ == "__main__":
    main()