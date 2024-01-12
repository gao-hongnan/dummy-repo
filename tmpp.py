from enum import Enum
from dataclasses import dataclass, field


@dataclass
class Reports:
    num_of_reports: int
    pages_per_report: int
    sentences_per_page: int
    words_per_sentence: int

    total_chunks: int = field(init=False)

    def calculate_total_chunks(self) -> int:
        """Using a fixed-size chunking strategy, calculate the total number of
        chunks needed given the number of reports, pages per report, sentences
        per page, and words per sentence.
        """
        return (
            self.num_of_reports
            * self.pages_per_report
            * self.sentences_per_page
            * self.words_per_sentence
        )

    def __post_init__(self) -> None:
        self.total_chunks = self.calculate_total_chunks()


class DataTypeSize(Enum):
    FLOAT16 = 2
    FLOAT32 = 4
    FLOAT64 = 8
    FLOAT128 = 16

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Embeddings:  # TextEmbeddings, ImageEmbeddings, AudioEmbeddings
    dimension: int
    data_type_size: DataTypeSize


@dataclass
class Metadata:
    pass


def calculate_storage_capacity(report: Reports, embeddings: Embeddings) -> int:
    """
    Calculate the storage capacity needed for a vector database.

    Args:
    params (CapacityParameters): Parameters including the number of chunks,
                                 their dimensionality, and the data type size.

    Returns:
    int: The total storage capacity required in bytes.
    """
    total_chunks = report.total_chunks
    storage_capacity_bytes = (
        total_chunks * embeddings.dimension * embeddings.data_type_size.value
    )

    return storage_capacity_bytes


report = Reports(
    num_of_reports=100,
    pages_per_report=250,
    sentences_per_page=100,
    words_per_sentence=15,
)
embeddings = Embeddings(dimension=768, data_type_size=DataTypeSize.FLOAT32)

storage_capacity = calculate_storage_capacity(report, embeddings)
print(storage_capacity)
