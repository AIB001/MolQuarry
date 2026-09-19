import httpx
import pytest

from molquarry import MolQuarry
from molquarry.http import HttpClient


@pytest.fixture
def make_quarry(tmp_path):
    instances = []

    def factory(handler, *, cache=False, **kwargs):
        http = HttpClient(
            transport=httpx.MockTransport(handler),
            throttle=False,
            cache_dir=tmp_path / "cache" if cache else None,
            sleep=lambda _: None,
            **kwargs,
        )
        instance = MolQuarry(home=tmp_path, http=http)
        instances.append(instance)
        return instance

    yield factory
    for instance in instances:
        instance.close()
