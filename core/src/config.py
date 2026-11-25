"""Typed configuration: one settings class per app and domain, each field an environment variable.

Every class derives from ``EchoraConfig`` and names its environment prefix:

.. code-block:: python

    class KaraConfig(EchoraConfig, env_prefix="ECHORA_KARA_"):
        model_id: str = compute("hcxlabs/echora-kara-700m-muq", legacy="FA_KARA_MODEL_ID")
        min_letter_ms: int = compute(40, ge=0, legacy="FA_KARA_MIN_LETTER_MS")
        startup_timeout_seconds: float = setting(120.0, gt=0)

- ``ECHORA_KARA_MIN_LETTER_MS`` sets ``min_letter_ms``; no code names it. A ``legacy`` name (the
  variable a field had before this class existed) is still read, after the canonical one.
- ``compute(...)`` marks a field that changes model results. ``compute_values()`` returns those
  fields, and ``with_compute_values()`` rebuilds a config from them: this is how a remote container
  computes with exactly the calling worker's settings.
- ``secret(...)`` fields are ``SecretStr``: hidden from ``repr`` and excluded from dumps.
- ``load()`` returns one validated, cached instance per class for the process;
  ``adopt()`` replaces it (a remote container adopting the caller's values) and
  ``reset_config_cache()`` clears every class's instance (tests that change the environment).

Values are validated when loaded, and validation errors never echo the input (it may be a
secret). Nothing reads ``.env`` files: Docker Compose passes them to the processes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Self

from pydantic import AliasChoices, Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ConfigEntry",
    "EchoraConfig",
    "compute",
    "config_classes",
    "reset_config_cache",
    "secret",
    "setting",
]

_LEGACY = "echora_legacy_env"
_COMPUTE = "echora_compute"


def setting(default: Any = ..., *, legacy: str | None = None, **constraints: Any) -> Any:
    """A configuration field. ``legacy`` is a previous environment name that is still read."""
    extra = {_LEGACY: legacy} if legacy else {}
    return Field(default, json_schema_extra=extra or None, **constraints)


def compute(default: Any = ..., *, legacy: str | None = None, **constraints: Any) -> Any:
    """A field that changes model results; remote computation receives its value."""
    extra: dict[str, Any] = {_COMPUTE: True}
    if legacy:
        extra[_LEGACY] = legacy
    return Field(default, json_schema_extra=extra, **constraints)


def secret(default: str = "", *, legacy: str | None = None, **constraints: Any) -> Any:
    """A secret field: declare it as ``SecretStr``. Never dumped or shown."""
    extra = {_LEGACY: legacy} if legacy else {}
    return Field(
        SecretStr(default),
        json_schema_extra=extra or None,
        repr=False,
        exclude=True,
        **constraints,
    )


def _extra(field: FieldInfo) -> dict[str, Any]:
    return field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}


@dataclass(frozen=True)
class ConfigEntry:
    """One environment variable of the settings catalog."""

    name: str
    field: str
    owner: str
    default: Any
    description: str
    secret: bool
    compute: bool
    deprecated_names: tuple[str, ...]


_registry: list[type[EchoraConfig]] = []


class EchoraConfig(BaseSettings):
    """Base class of every settings class. Subclasses set ``env_prefix`` as a class argument."""

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        hide_input_in_errors=True,
        case_sensitive=False,
        populate_by_name=True,
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
    )

    _instances: ClassVar[dict[type[EchoraConfig], EchoraConfig]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Read each field's legacy environment name after its canonical one.

        The bare field name is not a choice: ``REVISION`` or ``MODEL_ID`` in a container's
        environment must never configure a model. Code still passes fields by name
        (``populate_by_name``).
        """
        super().__pydantic_init_subclass__(**kwargs)
        prefix = cls.model_config.get("env_prefix", "")
        changed = False
        for name, field in cls.model_fields.items():
            legacy = _extra(field).get(_LEGACY)
            if legacy and field.validation_alias is None:
                field.validation_alias = AliasChoices(f"{prefix}{name}".upper(), legacy)
                changed = True
        if changed:
            cls.model_rebuild(force=True)
        _registry.append(cls)

    @classmethod
    def load(cls) -> Self:
        """The process's validated instance of this class, read from the environment once."""
        instance = EchoraConfig._instances.get(cls)
        if instance is None:
            instance = cls()
            EchoraConfig._instances[cls] = instance
        return instance  # type: ignore[return-value]

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> Self:
        """The config with these variables (by the same names) taking the place of the process's.

        A field ``environ`` does not name is still read from the process environment.
        """
        given = {name.upper(): value for name, value in environ.items()}
        values: dict[str, Any] = {}
        for entry in cls.entries():
            for name in (entry.name, *entry.deprecated_names):
                if name in given:
                    values[entry.field] = given[name]
                    break
        return cls.model_validate(values)

    @classmethod
    def adopt(cls, instance: Self) -> None:
        """Make ``instance`` what ``load()`` returns in this process (remote computation)."""
        if type(instance) is not cls:
            raise TypeError(f"Expected a {cls.__name__}, got {type(instance).__name__}")
        EchoraConfig._instances[cls] = instance

    @classmethod
    def compute_fields(cls) -> tuple[str, ...]:
        """Names of the fields that change model results."""
        return tuple(n for n, f in cls.model_fields.items() if _extra(f).get(_COMPUTE))

    def compute_values(self) -> dict[str, Any]:
        """This config's compute fields, as sent to remote computation."""
        names = self.compute_fields()
        return self.model_dump(include=set(names), mode="json")

    @classmethod
    def with_compute_values(cls, values: dict[str, Any]) -> Self:
        """The config with exactly the given compute fields, everything else as configured here.

        Raises ``ValueError`` for a name that is not a compute field, so a remote container never
        silently computes with a setting it did not receive.
        """
        unknown = set(values) - set(cls.compute_fields())
        if unknown:
            raise ValueError(f"Not compute fields of {cls.__name__}: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def entries(cls) -> Iterator[ConfigEntry]:
        """The environment variables this class reads, for the settings catalog."""
        prefix = cls.model_config.get("env_prefix", "")
        for name, field in cls.model_fields.items():
            alias = field.validation_alias
            if isinstance(alias, AliasChoices):
                names = [c for c in alias.choices if isinstance(c, str) and c != name]
            elif isinstance(alias, str):
                names = [alias]
            else:
                names = [f"{prefix}{name}".upper()]
            default = None if field.is_required() else field.get_default(call_default_factory=True)
            is_secret = field.annotation is SecretStr or field.exclude is True
            yield ConfigEntry(
                name=names[0].upper(),
                field=name,
                owner=cls.__name__,
                default=None if is_secret else default,
                description=field.description or "",
                secret=is_secret,
                compute=bool(_extra(field).get(_COMPUTE)),
                deprecated_names=tuple(n.upper() for n in names[1:]),
            )


def config_classes() -> list[type[EchoraConfig]]:
    """Every settings class defined so far (import the packages first), in definition order."""
    return list(_registry)


def reset_config_cache() -> None:
    """Forget every loaded instance, so the next ``load()`` reads the environment again."""
    EchoraConfig._instances.clear()
