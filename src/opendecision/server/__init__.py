"""Optional local HTTP API (install ``opendecision[server]``)."""

from .app import create_app

__all__ = ["create_app"]
