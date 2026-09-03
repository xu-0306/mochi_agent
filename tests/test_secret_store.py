from pathlib import Path
import pytest

from mochi.config.manager import (
    EMPTY_CONFIG_REVISION,
    config_revision,
    load_config,
    save_config,
)
import mochi.config.manager as config_manager
from mochi.config.schema import MochiConfig
from mochi.security.secret_store import SecretStoreError
from mochi.security.secret_store import SecretStore


def test_secret_store_round_trip_never_writes_plaintext(tmp_path: Path) -> None:
    store = SecretStore(tmp_path / "secrets.enc")
    reference = "config-v1:model:example"
    secret = "sk-test-secret-value"

    store.set(reference, secret)

    assert store.get(reference) == secret
    assert secret.encode("utf-8") not in (tmp_path / "secrets.enc").read_bytes()

    store.delete(reference)
    assert store.get(reference) is None


def test_secret_store_rejects_corrupt_ciphertext(tmp_path: Path) -> None:
    store = SecretStore(tmp_path / "secrets.enc")
    store.set("config-v1:model:example", "secret")
    (tmp_path / "secrets.enc").write_bytes(b"corrupt")

    with pytest.raises(SecretStoreError, match="Unable to read encrypted secrets"):
        store.get("config-v1:model:example")


def test_existing_store_without_protected_key_fails_closed(
    tmp_path: Path,
) -> None:
    store = SecretStore(tmp_path / "secrets.enc")
    store.set("config-v1:model:example", "secret")
    store.key_path.unlink()

    with pytest.raises(SecretStoreError, match="protected key is missing"):
        SecretStore(tmp_path / "secrets.enc")


def test_save_config_rolls_back_secret_store_when_yaml_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    original = MochiConfig.model_validate(
        {"openai_compat": {"api_key": "old-secret"}}
    )
    save_config(original, config_path, expected_revision=EMPTY_CONFIG_REVISION)
    before = (tmp_path / "secrets.enc").read_bytes()
    snapshot = load_config(config_path)
    replacement = snapshot.model_copy(update={
        "openai_compat": snapshot.openai_compat.model_copy(update={"api_key": "new-secret"})
    })

    def fail_yaml(*_args: object, **_kwargs: object) -> None:
        raise OSError("yaml write failed")

    monkeypatch.setattr("mochi.config.manager._atomic_write_config", fail_yaml)
    with pytest.raises(OSError, match="yaml write failed"):
        save_config(
            replacement,
            config_path,
            expected_revision=config_revision(config_path),
        )

    assert (tmp_path / "secrets.enc").read_bytes() == before
    assert load_config(config_path).openai_compat.api_key.get_secret_value() == "old-secret"


def test_save_config_restores_yaml_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    original_yaml = b"model: ollama:old\n"
    config_path.write_bytes(original_yaml)
    snapshot = load_config(config_path)
    updated = snapshot.model_copy(update={"model": "ollama:new"})

    def fail_fsync(_parent: Path) -> None:
        raise OSError("parent fsync failed")

    monkeypatch.setattr(config_manager, "_fsync_parent", fail_fsync)
    with pytest.raises(OSError, match="parent fsync failed"):
        save_config(
            updated,
            config_path,
            expected_revision=config_revision(config_path),
        )

    assert config_path.read_bytes() == original_yaml
    assert load_config(config_path).model == "ollama:old"


def test_legacy_secret_migration_restores_secret_store_when_yaml_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    original_yaml = b"openai_compat:\n  api_key: legacy-secret\n"
    config_path.write_bytes(original_yaml)

    def fail_yaml(*_args: object, **_kwargs: object) -> None:
        raise OSError("yaml cleanup failed")

    monkeypatch.setattr(config_manager, "_atomic_write_config", fail_yaml)
    loaded = load_config(config_path)

    assert loaded.openai_compat.api_key is not None
    assert loaded.openai_compat.api_key.get_secret_value() == "legacy-secret"
    assert config_path.read_bytes() == original_yaml
    assert not (tmp_path / "secrets.enc").exists()
    assert not (tmp_path / "secrets.enc.key").exists()


def test_save_config_clear_secret_scope_prevents_rehydration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    original = MochiConfig.model_validate(
        {"openai_compat": {"api_key": "old-secret"}}
    )
    save_config(original, config_path, expected_revision=EMPTY_CONFIG_REVISION)
    snapshot = load_config(config_path)
    cleared = snapshot.model_copy(update={
        "openai_compat": snapshot.openai_compat.model_copy(update={"api_key": None})
    })
    save_config(
        cleared,
        config_path,
        expected_revision=config_revision(config_path),
        clear_secret_scopes=["openai_compat/api_key"],
    )

    assert load_config(config_path).openai_compat.api_key is None
