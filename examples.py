from dataclasses import dataclass
from datetime import datetime
from utils import vectorize, TextType

@dataclass
class DocumentChunk:
    link: str | None
    max_num_tokens: int
    num_tokens: int
    chunk_index: int
    content: str
    embedding: list[float]

@dataclass
class DanswerDocument:
    document_id: str
    semantic_id: str
    title: str
    title_embedding: list[float]
    content: str
    chunks: list[DocumentChunk]
    source_type: str  # really an enum
    document_sets: list[str]
    # If multiple tags are applied they must ALL match including if the keys are different or multiple elements of the list
    metadata: dict[str, str | list[str]]
    boost_count: int
    last_updated: datetime | None
    hidden: bool


t1_title = "The weather in Florida"
t1_chunk1 = "The weather in Florida is hot and humid"
t1_chunk2 = "The weather in Alaska is frigid"
t1_chunk3 = "The weather in the Saharah is dry"

t2_title = "My favorite animal"
t2_chunk1 = "My favorite animal in the world is the dog"
t2_chunk2 = "My favorite animal in the world is the cat"
t2_chunk3 = "My favorite animal in Florida is the aligator"

t3_title = "The best food"
t3_chunk1 = "The best food is French fries"
t3_chunk2 = "The best food is pizza"
t3_chunk3 = "The best food is sushi"


TEST1 = DanswerDocument(
    document_id="test1",
    semantic_id="Test1",
    title=t1_title,
    title_embedding=vectorize(t1_title, TextType.PASSAGE),
    content="NA",
    chunks=[
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t1_chunk1.split()), chunk_index=0, content=t1_chunk1, embedding=vectorize(t1_chunk1, TextType.PASSAGE)),
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t1_chunk2.split()), chunk_index=1, content=t1_chunk2, embedding=vectorize(t1_chunk2, TextType.PASSAGE)),
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t1_chunk3.split()), chunk_index=2, content=t1_chunk3, embedding=vectorize(t1_chunk3, TextType.PASSAGE)),
    ],
    source_type="web",
    document_sets=["test_set"],
    metadata={"space": "HR"},
    boost_count=0,
    last_updated=datetime(2023, 9, 10),
    hidden=False,
)


TEST2 = DanswerDocument(
    document_id="test2",
    semantic_id="Test2",
    title=t2_title,
    title_embedding=vectorize(t2_title, TextType.PASSAGE),
    content="NA",
    chunks=[
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t2_chunk1.split()), chunk_index=0, content=t2_chunk1, embedding=vectorize(t2_chunk1, TextType.PASSAGE)),
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t2_chunk2.split()), chunk_index=1, content=t2_chunk2, embedding=vectorize(t2_chunk2, TextType.PASSAGE)),
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t2_chunk3.split()), chunk_index=2, content=t2_chunk3, embedding=vectorize(t2_chunk3, TextType.PASSAGE)),
    ],
    source_type="web",
    document_sets=["test_set"],
    metadata={"space": "HR"},
    boost_count=0,
    last_updated=datetime(2023, 9, 10),
    hidden=False,
)

TEST3 = DanswerDocument(
    document_id="test3",
    semantic_id="Test3",
    title=t3_title,
    title_embedding=vectorize(t3_title, TextType.PASSAGE),
    content="NA",
    chunks=[
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t3_chunk1.split()), chunk_index=0, content=t3_chunk1, embedding=vectorize(t3_chunk1, TextType.PASSAGE)),
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t3_chunk2.split()), chunk_index=1, content=t3_chunk2, embedding=vectorize(t3_chunk2, TextType.PASSAGE)),
        DocumentChunk(link=None, max_num_tokens=4096, num_tokens=len(t3_chunk3.split()), chunk_index=2, content=t3_chunk3, embedding=vectorize(t3_chunk3, TextType.PASSAGE)),
    ],
    source_type="web",
    document_sets=["test_set"],
    metadata={"space": "HR"},
    boost_count=0,
    last_updated=datetime(2023, 9, 10),
    hidden=False,
)

DOCUMENTS = [TEST1, TEST2, TEST3]

QUERY = "Florida"