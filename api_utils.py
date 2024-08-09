from enum import Enum, auto
import io
import logging
import os
from typing import Dict, IO, Optional, Tuple, cast

import flask
import requests
from requests import Response
from retry import retry

from placebo_api import config
from placebo_api.auth import auth
from placebo_api.auth.internal_api_auth import InternalApiAuth

_LOGGER = logging.getLogger(__name__)


class HTTPMethod(Enum):
    GET = auto()
    POST = auto()


def fetch_from_misc(
    endpoint: str,
    query_params: Optional[Dict[str, str]] = None,
    auth_info: Optional[InternalApiAuth] = None,
) -> IO[str]:
    misc_root = config.MISC_ROOT

    if auth_info is None:
        auth_info = _get_auth_info(basic_auth_env="MISC_AUTH")

    assert misc_root[-1] != "/"
    url = f"{misc_root}/{endpoint}"

    return _internal_api_call(url, query_params, auth_info, timeout_s=30)


def fetch_from_reflow(
    endpoint: str, query_params: Optional[Dict[str, str]] = None
) -> IO[str]:
    reflow_root = config.REFLOW_ROOT

    auth_info = _get_auth_info(basic_auth_env="REFLOW_AUTH")

    assert reflow_root[-1] != "/"
    url = f"{reflow_root}/{endpoint}"

    return _internal_api_call(url, query_params, auth_info, timeout_s=30)


def fetch_from_placebo(
    endpoint: str, query_params: Optional[Dict[str, str]] = None
) -> IO[str]:
    placebo_root = config.PLACEBO_ROOT
    auth_info = _get_auth_info(basic_auth_env="PLACEBO_API_AUTH")

    assert placebo_root[-1] != "/"
    url = f"{placebo_root}/{endpoint}"

    return _internal_api_call(url, query_params, auth_info, timeout_s=30)


def _get_auth_info(basic_auth_env: str) -> InternalApiAuth:
    if flask.has_request_context():
        user = auth.get_current_user()
        token = user.token
        assert token is not None

        return InternalApiAuth(token=token, basic=None)
    else:
        basic = _get_basic_auth(basic_auth_env)

        return InternalApiAuth(token=None, basic=basic)


def _get_basic_auth(env: str) -> Tuple[str, str]:
    user_and_pass = tuple(os.environ[env].split(":"))
    assert len(user_and_pass) == 2

    return cast(Tuple[str, str], user_and_pass)


class InternalApiError(Exception):
    def __init__(self, status_code: int, text: str) -> None:
        msg = f"Error {status_code}: {text}"
        super().__init__(msg)

        self.status_code = status_code
        self.text = text


def _internal_api_call_optional(
    url: str,
    query_params: Optional[Dict[str, str]],
    auth_info: InternalApiAuth,
    timeout_s: int,
    retries: int,
    status_for_none: Optional[int],
    method: HTTPMethod,
) -> Optional[IO[str]]:
    _LOGGER.info(
        "Fetching '%s' '%s'",
        url,
        query_params,
        extra={"action": "internal_api_call", "url": url, "query_params": query_params},
    )

    assert auth_info.basic is not None or auth_info.token is not None

    @retry(tries=retries, delay=1, backoff=2)
    def _call() -> Response:
        headers: Dict[str, str] = {}
        if auth_info.token is not None:
            headers["authorization"] = f"Bearer {auth_info.token}"

        if method == HTTPMethod.GET:
            res = requests.get(
                url=url,
                params=query_params,
                headers=headers,
                auth=auth_info.basic,
                timeout=timeout_s,
            )
        else:
            assert method == HTTPMethod.POST

            headers["content-type"] = "application/x-www-form-urlencoded"

            res = requests.post(
                url=url,
                data=query_params,
                headers=headers,
                auth=auth_info.basic,
                timeout=timeout_s,
            )

        if res.status_code >= 500:
            # retry 5xx errors
            raise InternalApiError(res.status_code, res.text)

        return res

    res = _call()

    if status_for_none is not None and res.status_code == status_for_none:
        return None

    if res.status_code != 200:
        raise InternalApiError(res.status_code, res.text)

    return io.StringIO(res.text)


def _internal_api_call(
    url: str,
    query_params: Optional[Dict[str, str]],
    auth_info: InternalApiAuth,
    timeout_s: int = 10,
    retries: int = 4,
    method: HTTPMethod = HTTPMethod.GET,
) -> IO[str]:
    res = _internal_api_call_optional(
        url,
        query_params,
        auth_info,
        timeout_s=timeout_s,
        retries=retries,
        status_for_none=None,
        method=method,
    )
    assert res is not None

    return res
