"""Conditional service/browser adapters; callers must supply explicit targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from ._optional import IntegrationUnavailable, require


def openapi_schema(document: Mapping[str, Any]) -> Any:
    """Load a supplied in-memory OpenAPI document through Schemathesis (no HTTP)."""
    def check_refs(value: Any) -> None:
        if isinstance(value, Mapping):
            if "$ref" in value and not str(value["$ref"]).startswith("#/"):
                raise ValueError("only document-local OpenAPI $ref values are allowed")
            for item in value.values():
                check_refs(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_refs(item)
    check_refs(document)
    framework = require("schemathesis")
    return framework.openapi.from_dict(dict(document))


def smoke_openapi(document: Mapping[str, Any], *, base_url: str, path: str,
                  method: str = "GET", case_kwargs: Mapping[str, Any] | None = None,
                  allow_network: bool = False, timeout: float = 10) -> dict[str, Any]:
    """Execute one explicitly configured GET/HEAD case and validate its response.

    For generated fuzz cases use ``openapi_schema(...).parametrize()`` in a
    dedicated pytest suite with an explicitly authorized service target.
    """
    if not allow_network:
        raise ValueError("service request requires allow_network=True and an explicit base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("base_url must be a HTTP(S) origin without embedded credentials")
    if method.upper() not in {"GET", "HEAD"}:
        raise ValueError("smoke_openapi only sends GET/HEAD; test mutations in a dedicated fixture")
    if "method" in (case_kwargs or {}):
        raise ValueError("case_kwargs cannot override the declared HTTP method")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    case = openapi_schema(document)[path][method.upper()].Case(**dict(case_kwargs or {}))
    response = case.call(base_url=base_url, timeout=timeout, allow_redirects=False)
    case.validate_response(response)
    return {"framework": "schemathesis", "status": "passed", "status_code": response.status_code,
            "method": method.upper(), "path": path, "cases_executed": 1}


def smoke_browser(*, html_path: str | Path | None = None, url: str | None = None,
                  allow_network: bool = False, expected_text: str | None = None,
                  timeout_ms: int = 10_000) -> dict[str, Any]:
    """Open a supplied local report or explicit URL in installed Chromium.

    Browser binaries are prerequisites and are never downloaded by this call.
    Local HTML runs with HTTP(S) requests blocked. Explicit URL mode permits
    only requests to the supplied origin and requires allow_network=True.
    """
    if (html_path is None) == (url is None):
        raise ValueError("provide exactly one html_path or url")
    if url is not None:
        origin = urlparse(url)
        if (not allow_network or origin.scheme not in {"http", "https"} or not origin.netloc
                or origin.username or origin.password):
            raise ValueError("URL mode requires HTTP(S) url and allow_network=True")
        target = url
    else:
        report = Path(html_path).expanduser().resolve()
        if not report.is_file():
            raise ValueError("html_path must be an existing file")
        origin = None
        target = report.as_uri()
    api = require("playwright.sync_api", "playwright plus a preinstalled Chromium browser")
    with api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except api.Error as exc:
            raise IntegrationUnavailable(f"Playwright Chromium is unavailable: {exc}") from exc
        try:
            context = browser.new_context(service_workers="block")

            def route_request(route: Any) -> None:
                parsed = urlparse(route.request.url)
                allowed = (parsed.scheme in {"file", "data", "about"} if origin is None
                           else (parsed.scheme, parsed.netloc) == (origin.scheme, origin.netloc))
                route.continue_() if allowed else route.abort()

            context.route("**/*", route_request)
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            response = page.goto(target, timeout=timeout_ms, wait_until="load")
            if response is not None and response.status >= 400:
                raise RuntimeError(f"browser target returned HTTP {response.status}")
            if expected_text is not None:
                page.get_by_text(expected_text, exact=False).first.wait_for(state="visible", timeout=timeout_ms)
            if errors:
                raise RuntimeError(f"browser page errors: {errors}")
            return {"framework": "playwright", "status": "passed", "title": page.title(),
                    "page_errors": errors, "target_kind": "url" if url else "local_report"}
        finally:
            browser.close()


def smoke_container(image: str, *, command: Sequence[str], check_command: Sequence[str] | None = None,
                    allow_start: bool = False, timeout: int = 30) -> dict[str, Any]:
    """Use Testcontainers with an already-local image and local Docker daemon.

    Nothing is pulled. Networking and the optional Ryuk helper are disabled;
    context/finally cleanup removes the container. A command that terminates is
    checked by exit code. For a running service, check_command runs inside it.
    A stopped daemon, absent image, or missing SDK is a prerequisite failure.
    """
    if not allow_start:
        raise ValueError("container execution requires allow_start=True with explicit image and command")
    if not image.strip() or not command or isinstance(command, str):
        raise ValueError("image and nonempty command argument list are required")
    if check_command is not None and (not check_command or isinstance(check_command, str)):
        raise ValueError("check_command must be a nonempty argument list")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    docker = require("docker", "testcontainers")
    try:
        client = docker.from_env(timeout=timeout)
        if client.api.base_url not in {"http+docker://localhost", "http+docker://localnpipe"}:
            raise ValueError("only a local Unix socket/named-pipe Docker daemon is permitted")
        client.ping()
        image_id = client.images.get(image).id
    except docker.errors.ImageNotFound as exc:
        raise IntegrationUnavailable(f"Docker image {image!r} is not local; no pull was attempted") from exc
    except docker.errors.DockerException as exc:
        raise IntegrationUnavailable(f"Local Docker is unavailable: {exc}") from exc
    finally:
        if "client" in locals():
            client.close()
    module = require("testcontainers.core.container", "testcontainers")
    config = require("testcontainers.core.config", "testcontainers")
    prior = config.testcontainers_config.ryuk_disabled
    config.testcontainers_config.ryuk_disabled = True
    container = None
    try:
        container = module.DockerContainer(
            image_id, network_disabled=True, use_config_proxy=False,
            docker_client_kw={"timeout": timeout},
        ).with_command(list(command))
        # Pin the inspected identity, irrespective of configured registry prefixes.
        container.image = image_id
        def no_pull(*args: Any, **kwargs: Any) -> None:
            raise IntegrationUnavailable("image pull blocked; preloaded image is required")
        container.get_docker_client().client.api.pull = no_pull
        container.start()
        wrapped = container.get_wrapped_container()
        if check_command:
            checked = wrapped.exec_run(list(check_command))
            exit_code = checked.exit_code
        else:
            exit_code = wrapped.wait(timeout=timeout)["StatusCode"]
        if exit_code != 0:
            raise RuntimeError(f"container smoke command exited with status {exit_code}")
        return {"framework": "testcontainers", "status": "passed", "image": image,
                "exit_code": exit_code, "image_pull_attempted": False, "network_enabled": False}
    finally:
        try:
            if container is not None:
                container.stop()
        finally:
            config.testcontainers_config.ryuk_disabled = prior
