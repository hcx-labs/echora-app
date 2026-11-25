"""Endpoints as classes, and categories that mount them.

An endpoint is one class in its own file: its method and path, its request and response models,
its dependencies, status code and OpenAPI metadata, and ``handle()``. A category lists its
endpoints; the v1 router mounts every category. Nothing registers routes on a global app.

``handle()`` is a plain FastAPI handler: path and query parameters, the request body (annotated
with the endpoint's ``request`` model) and dependencies are its parameters.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from fastapi import APIRouter, Depends
from fastapi.routing import APIRoute
from pydantic import BaseModel


class Endpoint:
    """One route: subclasses set the class attributes and implement ``handle()``."""

    method: ClassVar[str]
    path: ClassVar[str]
    """Relative to the category's prefix ("" for the prefix itself)."""
    request: ClassVar[type[BaseModel] | None] = None
    """The body model; ``handle()`` must take a parameter annotated with it."""
    response: ClassVar[Any] = None
    """The response model (``None``: no body, or a ``Response`` the handler builds)."""
    status_code: ClassVar[int | None] = None
    summary: ClassVar[str | None] = None
    """OpenAPI summary; the class docstring's first line by default."""
    name: ClassVar[str | None] = None
    """Route name for ``request.url_for``."""
    dependencies: ClassVar[Sequence[Callable[..., Any]]] = ()
    """Dependencies run for their effect (an admin check), not passed to ``handle()``."""
    response_class: ClassVar[type | None] = None
    exclude_unset: ClassVar[bool] = True
    """Leave out response fields the handler did not return, so optional keys stay absent
    rather than appear as null."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls) or "handle" not in cls.__dict__:
            return
        if cls.method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            raise TypeError(f"{cls.__name__}: unsupported method {cls.method}")
        if cls.request is not None:
            annotations = inspect.get_annotations(cls.handle, eval_str=True)
            if cls.request not in annotations.values():
                raise TypeError(f"{cls.__name__}.handle takes no {cls.request.__name__} body")

    def handle(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def register(cls, router: APIRouter) -> None:
        """Add this endpoint's route to ``router``."""
        options: dict[str, Any] = {}
        if cls.status_code is not None:
            options["status_code"] = cls.status_code
        if cls.response_class is not None:
            options["response_class"] = cls.response_class
        if cls.name is not None:
            options["name"] = cls.name
        doc = inspect.getdoc(cls) or ""
        router.add_api_route(
            cls.path,
            cls().handle,
            methods=[cls.method],
            response_model=cls.response,
            response_model_exclude_unset=cls.exclude_unset,
            summary=cls.summary or (doc.splitlines()[0] if doc else None),
            description=doc,
            dependencies=[Depends(dependency) for dependency in cls.dependencies],
            **options,
        )


@dataclass(frozen=True)
class Category:
    """A folder of endpoints under one prefix (``/library``, ``/search``)."""

    prefix: str
    endpoints: Sequence[type[Endpoint]]
    tags: Sequence[str] = ()
    route_class: type[APIRoute] = APIRoute
    """For example a route class that redacts secrets from validation errors."""
    dependencies: Sequence[Callable[..., Any]] = field(default_factory=tuple)

    def router(self) -> APIRouter:
        """Every endpoint of the category, mounted under its prefix."""
        router = APIRouter(
            prefix=self.prefix,
            tags=list(self.tags) or [self.prefix.strip("/") or "root"],
            route_class=self.route_class,
            dependencies=[Depends(dependency) for dependency in self.dependencies],
        )
        for endpoint in self.endpoints:
            endpoint.register(router)
        return router
