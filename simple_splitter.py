from __future__ import annotations
from langchain.text_splitter import TextSplitter
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import re

class EmptyLineSplitter(TextSplitter):
    def __init__(self, **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        return re.split(r'\n{2,}',text)
