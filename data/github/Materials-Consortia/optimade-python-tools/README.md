# https://github.com/Materials-Consortia/optimade-python-tools

```console
tests/server/utils.py:class AsyncHttpxTestClient(httpx.AsyncClient):
optimade/client/client.py:    _http_client: type[httpx.AsyncClient] | type[requests.Session] | None = None
optimade/client/client.py:        http_client: None | (type[httpx.AsyncClient] | type[requests.Session]) = None,
optimade/client/client.py:            if issubclass(self._http_client, httpx.AsyncClient):
optimade/client/client.py:                self._http_client = httpx.AsyncClient

```
