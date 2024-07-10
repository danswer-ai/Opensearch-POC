from datetime import datetime

from opensearchpy import OpenSearch
from opensearchpy.helpers.document import Document, InnerDoc
from opensearchpy.helpers.field import Text, Double, Nested, Date, DenseVector
from opensearchpy import Search

from examples import DOCUMENTS, DanswerDocument, QUERY

from utils import vectorize, TextType, EMBEDDING_DIM


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
                "content": {"type": "text", "index": False},  # All keyword contents are stored at the "chunk" level
                "title_vector": hnsw_config,
                "chunks": {
                    "type": "nested",
                    "properties": {
                        "link": {"type": "text", "index": False},  # Nullable by default
                        "max_num_tokens": {"type": "integer", "index": False, "null_value": 512},
                        "num_tokens": {"type": "integer", "index": False},
                        "chunk_index": {"type": "integer", "index": False},
                        "content": {"type": "text"},
                        "embedding": hnsw_config
                    }
                },
                "source_type": {"type": "keyword"},
                "document_sets": {"type": "keyword"},
                "metadata": {
                    "type": "nested",
                    "properties": {
                        "key": {"type": "keyword"},
                        # The value can be a single term or an array (just add multiple to it at indexing time)
                        "value": {"type": "keyword"}
                    }
                },
                "last_updated": {"type": "date"},
                # 0 default, positive for upvoted, negative for downvoted
                "boost_count": {"type": "integer", "null_value": 0},
                "not_hidden": {"type": "boolean", "null_value": True},
            }
        }
    }

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
                                # 0.4,  # Keyword BM25 which includes title and content
                                # 0.5,  # Title Vector Boost (Title already included in other chunks)
                                1.0   # Chunk Vector score across different chunk sizes
                            ]
                        }
                    }
                }
            }
        ]
    }
    client.search_pipeline.put(id="normalization_step", body=pipeline_body)

def index_document(client, index_name, document: Document):
    def _expand_dict(dict):
        return [
            {"key": k, "value": v} for k, v in dict.items()
        ]

    doc = {
        "title": document.title,
        "content": document.content,
        "title_vector": vectorize(document.title, TextType.PASSAGE),
        "chunks": [
            {
                "link": chunk.link,
                "max_num_tokens": chunk.max_num_tokens,
                "num_tokens": chunk.num_tokens,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": vectorize(chunk.content, TextType.PASSAGE)
            } for chunk in document.chunks
        ],
        "metadata": _expand_dict(document.metadata),
        "source_type": "web",
        "document_sets": document.document_sets,
        "last_updated": document.last_updated,
        "boost_count": 0,
        "not_hidden": not document.hidden
    }

    print(f"Indexing {document.title} document")
    response = client.index(index=index_name, body=doc)
    print(response)

def hybrid_search(client, index_name, query, max_num_results=10):
    # TODO ADD ACL
    start_date_str = datetime(2023, 11, 14).isoformat()
    end_date_str = datetime(2023, 11, 16).isoformat()
    query_vector = vectorize(query, TextType.QUERY)

    # https://opensearch.org/docs/latest/search-plugins/search-pipelines/normalization-processor/#search-tuning-recommendations
    search_body_complete = {
        "size": max_num_results,  # Number of results to return
        "query": {
            "bool": {
                "must": [
                    {
                        "function_score": {
                            "query": {
                                "function_score": {
                                    "query": {
                                        "nested": {
                                            "path": "chunks",
                                            "query": {
                                                # Apply the Normalization pipeline step for each result within the queries of hybrid
                                                "hybrid": {
                                                    "queries": [
                                                        # Keyword score that includes both the overall document title and the chunk content
                                                        {
                                                            # This may still be filtering out all the docs, need to verify
                                                            "multi_match": {
                                                                "query": query,
                                                                "type": "most_fields",
                                                                "fields": [
                                                                    "title^1.2",  # Means it"s boosted
                                                                    "chunks.content"
                                                                ],
                                                                "_name": "combined_keyword_score",
                                                            },
                                                        },
                                                        # Title Vector Score
                                                        {
                                                            "knn": {
                                                                "title_vector": {
                                                                    "vector": query_vector,
                                                                    "k": max_num_results,  # Similar to the size but it"s per shard (in this case we only have one)
                                                                    "_name": "title_vector_score"
                                                                },
                                                            },
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
                                            "inner_hits": {
                                                "size": 20
                                            },
                                        },
                                    },
                                    "functions": [
                                        {
                                            "filter": {"exists": {"field": "last_updated"}},
                                            "gauss": {
                                                "last_updated": {
                                                    "origin": "now",
                                                    # Reaches 0.5 score at 1 year (but this is 0.5 of the 0.25)
                                                    # Slower decay over time eventually
                                                    "scale": "365d",
                                                    "decay": 0.5  # This could be set at query time to affect the rate of decay
                                                }
                                            },
                                            "weight": 0.25
                                        },
                                        {
                                            # If no date, then give a ~1/4th penalty (like 1 quarter old)
                                            # Note that if we modify the decay rate, this may not be approximate to 1 quarter anymore
                                            "filter": {"bool": {"must_not": {"exists": {"field": "last_updated"}}}},
                                            "weight": 0.18
                                        },
                                        # Always add a 0.75 capping the decay
                                        {
                                            "weight": 0.75
                                        }
                                    ],
                                    "score_mode": "sum",
                                    "boost_mode": "multiply"
                                },
                            },
                            # 0.5 to 2x score: piecewise sigmoid function stretched out by factor of 3
                            # meaning requires 3x the number of feedback votes to have default sigmoid effect
                            "script_score": {
                                "script": {
                                    "source": """
                                    double boost_count = doc["boost_count"].value;
                                    if (boost_count < 0) {
                                        return 0.5 + (1 / (1 + Math.exp(-boost_count / 3)));
                                    } else {
                                        return 2 / (1 + Math.exp(-boost_count / 3));
                                    }
                                    """,
                                    "lang": "painless"
                                }
                            },
                            "boost_mode": "multiply"
                        },
                    },
                ],
                "filter": [
                    # More efficient as filters are cached where as must_not is not
                    # Only boolean scores, likely just more efficient overall
                    {
                        "term": {"not_hidden": True}
                    },
                    {
                        "range": {
                            "last_updated": {
                                "gte": start_date_str,
                                "lte": end_date_str
                            }
                        }
                    },
                    {
                        "terms": {
                            # This does an "or" of the values
                            # for matching every one, it would be 3 lines with single value rather than array
                            "document_sets": ["set1", "set2"]
                        }
                    },
                    {
                        "nested": {
                            "path": "metadata",
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"metadata.key": "space"}},
                                        {"term": {"metadata.value": "IT"}}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "nested": {
                            "path": "metadata",
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"metadata.key": "space"}},
                                        {"term": {"metadata.value": "HR"}}
                                    ]
                                }
                            }
                        }
                    }
                ],
            },
        },
    }


    # Gets one chunk per doc for some reason
    # Normalization is not applied as expected, not sure if applied at all (it's not applied across the same chunks of the doc either), no clue what is going on
    # Can get inner hits but it's just one chunk per hit as well
    search_body_hybrid_inside = {
        "size": max_num_results,  # Number of results to return
        "query": {
            "nested": {
                "path": "chunks",
                "query": {
                    "hybrid": {
                        "queries": [
                            # Keyword score that includes both the overall document title and the chunk content
                            # {
                            #     "multi_match": {
                            #         "query": query,
                            #         "fields": [
                            #             "title^1.2",  # Means it's boosted
                            #             "chunks.content"
                            #         ],
                            #         "_name": "combined_keyword_score",
                            #     },
                            # },
                            # Title Vector Score
                            # {
                            #     "knn": {
                            #         "title_vector": {
                            #             "vector": query_vector,
                            #             "k": max_num_results,
                            #             "_name": "title_vector_score"
                            #         },
                            #     },
                            # },
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

    
    # The sections seem to be getting the normalization applied correctly
    # However with just the chunk vectors, the calculations aren't correct
    # Checked:
    # - if the normalization for the chunks is done by taking into all chunks irrespective of the doc
    # - if the normalization for the chunks is done by taking the max chunk score of each document
    # Unverified: whether the scores are correct for the chunk or mixing the max scores of each query (meaning mixing chunks)
    search_body_hybrid_outside = {
        "size": max_num_results,  # Number of results to return
        "query": {
            "hybrid": {
                "queries": [
                    # Keyword score that includes both the overall document title and the chunk content
                    # {
                    #     "nested": {
                    #         "path": "chunks",
                    #         "query": {
                    #             "match": {
                    #                 "chunks.content": {
                    #                     "query": query,
                    #                     "_name": "content_keyword_score"
                    #                 }
                    #             }
                    #         },
                    #         "score_mode": "max",
                    #         "inner_hits": {
                    #             "size": 20
                    #         }
                    #     },
                    # },
                    # Title Vector Score
                    # {
                    #     "knn": {
                    #         "title_vector": {
                    #             "vector": query_vector,
                    #             "k": max_num_results,
                    #             "_name": "title_vector_score"
                    #         },
                    #     },
                    # },
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

    search_body_content_only = {
        "size": max_num_results,  # Number of results to return
        "query": {
            "nested": {
                "path": "chunks",
                "query": {
                    "match": {
                        "chunks.content": query,
                    }
                },
            },
        }
    }

    search_body_all = {
        "size": max_num_results,  # Number of results to return
        "query": {
            "match_all": {}
        }
    }
    
    response = client.search(
        index=index_name,
        search_pipeline="normalization_step",
        body=search_body_hybrid_outside,
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
    print(hybrid_search(client, index_name, QUERY))

if __name__ == "__main__":
    main()
