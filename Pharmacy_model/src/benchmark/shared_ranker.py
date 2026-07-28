from __future__ import annotations

from functools import lru_cache

from src.benchmark.product_ranker import ProductRanker


@lru_cache(maxsize=1)
def get_shared_product_ranker() -> ProductRanker:
    return ProductRanker()


__all__ = ["get_shared_product_ranker"]
