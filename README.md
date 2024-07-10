# Opensearch Nested Hybrid Search POC

This repo contains a minimal reproduction of the issues with hybrid search with nested fields.

For the basic example, see `basic_example.py`

We are evaluating against the latest version:
```
docker run -d \
  --name danswer_opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=D@nswer_1ndex" \
  opensearchproject/opensearch:latest
```


The gist of the problem is:
To be able to do RAG well, often times it's useful to find specific sections of documents to pass to the LLM.
Passing in whole documents has problems such as
- cost
- less accurate retrieval at scale due to inability to do hybrid search on chunks
- lost context since many embedding models don't handle massive context
- higher probability of LLM hallucinating or picking up on wrong sections/documents

A temporary workaround of making chunks as documents in opensearch has many other downsides
- latency/complexity, fetching a full document or adjacent chunks is now multiple calls
- cases where retrieving 50 hits could actually be less than 50 as multiple of them are chunks of the same doc
- some pipelines work at a document level and doesn't support this hack of treating chunks as documents
- updates to any field in the actual document is now reflected as a change in multiple opensearch documents
- implementation complexity and parallelization may cause unexpected issues that we don't fully see yet


This seems to be a common request, mentioned in a couple recent issues including:
https://github.com/opensearch-project/neural-search/issues/718
https://github.com/opensearch-project/ml-commons/issues/2612
