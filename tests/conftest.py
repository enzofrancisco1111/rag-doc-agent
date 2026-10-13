import importlib

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["CHROMA_DIR"] = str(tmp_path_factory.mktemp("chroma"))
    os.environ["ANTHROPIC_API_KEY"] = ""  # vide : les tests restent hors-ligne même si un .env existe
    from app import config

    importlib.reload(config)
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as c:
        yield c
