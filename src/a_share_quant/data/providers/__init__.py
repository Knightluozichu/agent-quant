"""Data providers package.

Provides unified data access through the DataProvider protocol.
"""

from a_share_quant.data.providers.base import BaseDataProvider, DataProvider
from a_share_quant.data.providers.mock import MockProvider
from a_share_quant.data.providers.local_parquet import LocalParquetProvider

__all__ = [
    "BaseDataProvider",
    "DataProvider",
    "MockProvider",
    "LocalParquetProvider",
    "get_data_provider",
]


def get_data_provider(provider_type: str = "mock", **kwargs: object) -> DataProvider:
    """Factory function to get a data provider.

    Args:
        provider_type: Type of provider ("mock", "local_parquet", "joinquant")
        **kwargs: Additional arguments for the provider

    Returns:
        DataProvider instance
    """
    if provider_type == "mock":
        return MockProvider(**kwargs)  # type: ignore[arg-type]

    if provider_type == "local_parquet":
        return LocalParquetProvider(**kwargs)  # type: ignore[arg-type]

    if provider_type == "joinquant":
        from a_share_quant.data.providers.joinquant import JoinQuantProvider

        return JoinQuantProvider(**kwargs)  # type: ignore[arg-type]

    msg = f"Unknown provider type: {provider_type}"
    raise ValueError(msg)
