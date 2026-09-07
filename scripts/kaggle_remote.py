"""Private, project-scoped Kaggle execution with local-only credentials.

``prepare`` is offline. ``upload-runtime`` and ``push`` are explicit network
mutations; ``status``, ``logs`` and ``output`` inspect the staged kernel only.
``watch`` waits for that kernel, downloads its artifacts and archives them as a
private Kaggle Dataset. It never launches training or edits another project.

Each command is a fresh process, so KAGGLE_CONFIG_DIR and the optional vendor SDK
path cannot change another project's credentials or Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "neuron-death-fuzzy-v1-"
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "FAILED"}
FORBIDDEN = {".git", ".codex", ".claude", ".agents", "api_kaggle", "paper", "remote_runs"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def scoped_path(path: Path, parent: Path = ROOT / "remote_runs") -> Path:
    path, parent = path.resolve(), parent.resolve()
    if not path.is_relative_to(parent) or path == parent:
        raise ValueError("Remote artifacts must be in a named subdirectory of remote_runs/.")
    return path


def read_credentials(directory: Path) -> dict:
    path = directory.resolve() / "kaggle.json"
    if not directory.resolve().is_relative_to(ROOT):
        raise ValueError("Use this project's credential directory.")
    # Intentionally never include parser details or contents in an exception.
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", data.get("username", "")) or not data.get("key"):
            raise ValueError
    except (OSError, ValueError, TypeError):
        raise ValueError("Project kaggle.json must contain a valid username and nonempty key.") from None
    return data


def api_client(credentials_dir: Path, sdk_path: Path | None):
    credentials = read_credentials(credentials_dir)
    # Remove inherited account/token and debug settings. The current process is
    # dedicated to this command; no persistent shell/user environment is edited.
    for key in tuple(os.environ):
        if key.startswith("KAGGLE_"):
            os.environ.pop(key)
    os.environ["KAGGLE_CONFIG_DIR"] = str(credentials_dir.resolve())
    os.environ["KAGGLE_USERNAME"] = credentials["username"]
    os.environ["KAGGLE_KEY"] = credentials["key"]
    os.environ["KAGGLE_ENABLE_OAUTH"] = "false"
    if sdk_path:
        if not (sdk_path / "kaggle" / "api" / "kaggle_api_extended.py").is_file():
            raise ValueError("SDK path does not contain the Kaggle Python client.")
        sys.path.insert(0, str(sdk_path.resolve()))
    if importlib.util.find_spec("kaggle") is None:
        raise RuntimeError("Install kaggle or pass --sdk-path to an existing client directory.")
    from kaggle import api
    if api.get_config_value(api.CONFIG_NAME_USER) != credentials["username"]:
        raise RuntimeError("Kaggle client authenticated with an unexpected account.")
    return api


def safe_archive_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(
        name and not path.is_absolute() and ".." not in path.parts
        and "\\" not in name and ":" not in name
        and not any(p.lower() in FORBIDDEN for p in path.parts)
        and path.name.lower() not in {"kaggle.json", "agents.md", "claude.md"}
        and not path.name.lower().startswith(".env")
        and path.suffix.lower() not in {".tex", ".pem", ".key"}
    )


def archive_hashes(payload: Path) -> dict[str, str]:
    with zipfile.ZipFile(payload) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)) or any(not safe_archive_member(n) for n in names):
            raise ValueError("Runtime archive contains an unsafe/private path or duplicate member.")
        if not {"src/train.py", "data/mnist.npz"}.issubset(names):
            raise ValueError("Runtime archive must contain src/train.py and data/mnist.npz.")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def bootstrap(dataset_ref: str, hashes: dict[str, str], payload_hash: str) -> str:
    owner, slug = dataset_ref.split("/")
    return f'''# Generated source verification. No credentials are stored in this notebook.
import hashlib, os, sys, zipfile
from pathlib import Path
_roots = [p for p in [Path("/kaggle/input/{slug}"),
    Path("/kaggle/input/datasets/{owner}/{slug}")] if p.is_dir()]
assert len(_roots) == 1, f"Expected exactly one attached runtime, found {{_roots}}"
_mount = _roots[0]
_target = Path("/kaggle/working/neuron-death-source")
_target.mkdir(parents=True, exist_ok=True)
_hashes = {hashes!r}
_archive = _mount / "runtime.zip"
if _archive.is_file():
    assert hashlib.sha256(_archive.read_bytes()).hexdigest() == {payload_hash!r}
    with zipfile.ZipFile(_archive) as _zip:
        for _name, _expected in _hashes.items():
            _data = _zip.read(_name)
            assert hashlib.sha256(_data).hexdigest() == _expected, _name
            _dest = (_target / _name).resolve()
            assert _dest.is_relative_to(_target.resolve()), _name
            _dest.parent.mkdir(parents=True, exist_ok=True)
            _dest.write_bytes(_data)
else:
    _expanded = [p for p in [_mount / "runtime", _mount] if (p / "src/train.py").is_file()]
    assert len(_expanded) == 1, "Expanded runtime not found"
    for _name, _expected in _hashes.items():
        _data = (_expanded[0] / _name).read_bytes()
        assert hashlib.sha256(_data).hexdigest() == _expected, _name
        _dest = (_target / _name).resolve()
        assert _dest.is_relative_to(_target.resolve()), _name
        _dest.parent.mkdir(parents=True, exist_ok=True)
        _dest.write_bytes(_data)
os.environ["NEURON_DEATH_SOURCE"] = str(_target)
os.environ["NEURON_DEATH_DATA"] = str(_target / "data")
os.chdir(_target)
sys.path.insert(0, str(_target))
print("NEURON_DEATH_SOURCE_VERIFIED", {payload_hash!r}, flush=True)
'''


def prepare(
    notebook: Path,
    payload: Path,
    label: str,
    username: str,
    accelerator: str = "gpu",
    reuse_runtime_dataset: str | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid username.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,17}", label):
        raise ValueError("Label must be 1–18 lower-case letters, digits or dashes.")
    hashes = archive_hashes(payload)
    payload_hash = digest(payload)
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook_hash = hashlib.sha256(json.dumps(nb, sort_keys=True).encode()).hexdigest()
    identity = hashlib.sha256((payload_hash + notebook_hash).encode()).hexdigest()[:8]
    slug = PREFIX + label + "-" + identity
    if reuse_runtime_dataset is not None:
        # The account token may allow versioning an owned Dataset but not
        # datasets.create.  Reuse is deliberately restricted to this project's
        # established code Dataset; an arbitrary owner/slug is never accepted.
        if reuse_runtime_dataset != username + "/neuron-death-code":
            raise ValueError("Only this account's neuron-death-code Dataset may be reused.")
        dataset_ref = reuse_runtime_dataset
        dataset_slug = "existing-neuron-death-code-" + payload_hash[:12]
        runtime_mode = "version_existing"
    else:
        dataset_slug = "neuron-death-fuzzy-runtime-" + payload_hash[:12]
        dataset_ref = username + "/" + dataset_slug
        runtime_mode = "create_immutable"
    run_dir = scoped_path(ROOT / "remote_runs" / slug)
    if run_dir.exists():
        raise FileExistsError("Run staging already exists; use its manifest or a new label.")
    dataset_dir = scoped_path(ROOT / "remote_runs" / "runtime" / dataset_slug)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_zip = dataset_dir / "runtime.zip"
    if target_zip.exists() and digest(target_zip) != payload_hash:
        raise RuntimeError("Immutable runtime directory already contains different bytes.")
    shutil.copyfile(payload, target_zip)
    write_json(dataset_dir / "dataset-metadata.json", {
        "id": dataset_ref,
        "title": ("neuron-death-code" if runtime_mode == "version_existing"
                  else "Neuron Death Fuzzy Runtime " + payload_hash[:12]),
        "licenses": [{"name": "other"}],
    })
    write_json(dataset_dir / "runtime-manifest.json", {
        "runtime_sha256": payload_hash, "files": hashes,
    })
    nb["cells"].insert(0, {
        "cell_type": "code", "execution_count": None, "outputs": [],
        "metadata": {}, "id": "verified-runtime",
        "source": bootstrap(dataset_ref, hashes, payload_hash).splitlines(keepends=True),
    })
    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True)
    code_file = kernel_dir / "fuzzy_learning.ipynb"
    write_json(code_file, nb)
    if code_file.stat().st_size >= 1_000_000:
        raise ValueError("Staged notebook exceeds Kaggle's one-megabyte source limit.")
    kernel_ref = username + "/" + slug
    write_json(kernel_dir / "kernel-metadata.json", {
        "id": kernel_ref, "title": slug.replace("-", " "),
        "code_file": code_file.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": accelerator == "gpu", "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4" if accelerator == "gpu" else "",
        "dataset_sources": [dataset_ref], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    })
    manifest = run_dir / "manifest.json"
    write_json(manifest, {
        "schema_version": 1, "created_utc": utc_now(), "kernel": kernel_ref,
        "url": "https://www.kaggle.com/code/" + kernel_ref,
        "runtime_dataset": dataset_ref, "runtime_directory": str(dataset_dir),
        "runtime_mode": runtime_mode,
        "runtime_sha256": payload_hash, "source_notebook_sha256": notebook_hash,
        "staged_notebook_sha256": digest(code_file), "kernel_directory": str(kernel_dir),
        "results_dataset": username + "/" + slug + "-results",
        "state": "PREPARED", "version": None,
    })
    return manifest


def read_manifest(path: Path, username: str) -> dict:
    path = scoped_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("kernel", "results_dataset"):
        if not data[key].startswith(username + "/" + PREFIX):
            raise ValueError("Manifest points outside this project's private resource namespace.")
    allowed_runtime = (
        data["runtime_dataset"].startswith(username + "/neuron-death-fuzzy-runtime-")
        or data["runtime_dataset"] == username + "/neuron-death-code"
    )
    if not allowed_runtime:
        raise ValueError("Unexpected runtime Dataset.")
    for key in ("runtime_directory", "kernel_directory"):
        scoped_path(Path(data[key]))
    return data


def assert_response_ok(response) -> None:
    error = getattr(response, "error", "") or getattr(response, "error_message", "")
    if error:
        raise RuntimeError(str(error))


def upload_runtime(api, manifest: dict, manifest_path: Path) -> str:
    directory = Path(manifest["runtime_directory"])
    if digest(directory / "runtime.zip") != manifest["runtime_sha256"]:
        raise RuntimeError("Runtime payload changed after staging.")
    if manifest.get("runtime_mode") == "version_existing":
        if manifest.get("runtime_version_attempted_utc"):
            raise RuntimeError("Runtime version submission was already attempted; inspect Dataset status.")
        state = api.dataset_status(manifest["runtime_dataset"])
        if str(state).lower() not in {"ready", "complete"}:
            raise RuntimeError("Existing runtime Dataset is not ready: " + str(state))
        metadata = json.loads((directory / "dataset-metadata.json").read_text(encoding="utf-8"))
        if metadata.get("id") != manifest["runtime_dataset"]:
            raise RuntimeError("Runtime Dataset metadata changed after staging.")
        # Write-before-call makes an uncertain network response non-repeatable.
        manifest["runtime_version_attempted_utc"] = utc_now()
        write_json(manifest_path, manifest)
        response = api.dataset_create_version(
            str(directory),
            version_notes="Fuzzy v1 runtime " + manifest["runtime_sha256"][:12],
            quiet=True,
            convert_to_csv=False,
            delete_old_versions=False,
        )
        assert_response_ok(response)
        manifest["runtime_version"] = getattr(response, "version_number", None)
        manifest["state"] = "RUNTIME_UPLOADED"
        write_json(manifest_path, manifest)
        return "versioned"
    try:
        return api.dataset_status(manifest["runtime_dataset"])
    except Exception as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) != 404:
            raise
    response = api.dataset_create_new(str(directory), public=False, quiet=True, convert_to_csv=False)
    assert_response_ok(response)
    return "created"


def owned_kernel_exists(api, kernel_ref: str) -> bool:
    """Exact duplicate guard for clients that map a missing private kernel to 403."""
    _, slug = kernel_ref.split("/", 1)
    candidates = api.kernels_list(mine=True, search=slug, page=1, page_size=100) or []
    return any(getattr(item, "ref", None) == kernel_ref for item in candidates)


def push(api, manifest: dict, manifest_path: Path) -> dict:
    if manifest.get("version") is not None or manifest.get("push_attempted_utc"):
        raise RuntimeError("This kernel was already submitted or submission is uncertain; inspect status.")
    directory = Path(manifest["kernel_directory"])
    meta = json.loads((directory / "kernel-metadata.json").read_text())
    if (meta["id"] != manifest["kernel"] or meta.get("is_private") is not True
            or meta.get("enable_internet") is not False
            or meta.get("dataset_sources") != [manifest["runtime_dataset"]]):
        raise RuntimeError("Staged private kernel metadata changed.")
    if digest(directory / meta["code_file"]) != manifest["staged_notebook_sha256"]:
        raise RuntimeError("Staged notebook changed.")
    # A globally unique slug still receives a read-before-write guard: never
    # create a new version of an existing notebook, even one from this project.
    try:
        api.kernels_status(manifest["kernel"])
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        missing_private = (
            isinstance(exc, ValueError)
            and "Permission 'kernels.get' was denied" in str(exc)
            and not owned_kernel_exists(api, manifest["kernel"])
        )
        if status_code != 404 and not missing_private:
            raise
    else:
        raise RuntimeError("Kernel already exists; refusing to overwrite or launch another version.")
    runtime_status = api.dataset_status(manifest["runtime_dataset"])
    if str(runtime_status).lower() not in {"ready", "complete"}:
        raise RuntimeError("Runtime Dataset is not ready: " + str(runtime_status))
    manifest["push_attempted_utc"] = utc_now()
    write_json(manifest_path, manifest)
    response = api.kernels_push(str(directory), timeout="39600",
                                acc="NvidiaTeslaT4" if meta["enable_gpu"] else None)
    assert_response_ok(response)
    if response.ref != manifest["kernel"]:
        raise RuntimeError("Server returned a different kernel reference; inspect before retrying.")
    manifest.update(version=response.version_number, kernel_id=response.kernel_id,
                    state="SUBMITTED", submitted_utc=utc_now())
    write_json(manifest_path, manifest)
    return {"kernel": manifest["kernel"], "version": response.version_number, "url": manifest["url"]}


def status(api, manifest: dict) -> dict:
    result = api.kernels_status(manifest["kernel"])
    name = getattr(result.status, "name", str(result.status)).split(".")[-1]
    return {"status": name, "failure_message": result.failure_message, "checked_utc": utc_now()}


def logs(api, manifest: dict, output_dir: Path, seconds: int = 20) -> str:
    """Read the official live endpoint; fall back to persisted execution logs."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
    owner, slug = manifest["kernel"].split("/")
    request = ApiGetKernelSessionLogsStreamRequest()
    request.user_name, request.kernel_slug = owner, slug
    request.version_label = str(manifest.get("version") or "latest")
    request.wait_for_logs_url_seconds = min(20, seconds)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    try:
        with api.build_kaggle_client() as client:
            # The endpoint returns requests.Response (FileDownload). A read
            # timeout keeps an idle log stream from blocking the local monitor.
            http = client.kernels.kernels_api_client._client
            original = http._prepare_response
            # Only this one request uses streamed transport; response lifetime
            # stays inside the client context below.
            response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
            started = time.monotonic()
            for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if line:
                    lines.append(line if isinstance(line, str) else line.decode("utf-8", "replace"))
                if time.monotonic() - started >= seconds or (lines and "END_OF_LOG" in lines[-1]):
                    break
            response.close()
    except Exception:
        # The public client may not support a live session yet; never equate
        # kernel RUNNING with evidence that the training loop has started.
        api.kernels_output(manifest["kernel"], str(output_dir), file_pattern=r"(?!)", quiet=True)
        lines = [p.read_text(encoding="utf-8", errors="replace") for p in output_dir.glob("*.log")]
    text = "\n".join(lines)
    if text:
        (output_dir / "live.log").write_text(text, encoding="utf-8")
    return text


def download_outputs(api, manifest: dict, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Kaggle returns generated artifacts, never a local folder selected by a
    # remote filename. The SDK output downloader itself handles nested paths.
    paths, token = api.kernels_output(manifest["kernel"], str(output_dir), quiet=True)
    if token:
        raise RuntimeError("Output pagination is present; do not mark this download complete.")
    files = [Path(p) for p in paths]
    for file in files:
        if not file.resolve().is_relative_to(output_dir.resolve()):
            raise RuntimeError("Downloaded path escaped the scoped output directory.")
    return [str(p.relative_to(output_dir)) for p in files]


def archive_outputs(api, manifest: dict, run_dir: Path) -> dict:
    outputs = run_dir / "output"
    archive_dir = run_dir / "results_dataset"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / "kernel-output.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(outputs.rglob("*")):
            if file.is_file():
                z.write(file, file.relative_to(outputs).as_posix())
    write_json(archive_dir / "provenance.json", {
        "kernel": manifest["kernel"], "kernel_version": manifest["version"],
        "runtime_sha256": manifest["runtime_sha256"], "archived_utc": utc_now(),
        "archive_sha256": digest(archive), "terminal_state": manifest.get("last_status"),
    })
    write_json(archive_dir / "dataset-metadata.json", {
        "id": manifest["results_dataset"],
        "title": manifest["results_dataset"].split("/")[1].replace("-", " "),
        "licenses": [{"name": "other"}],
    })
    # This resource belongs to this run alone; create once, never make public.
    try:
        state = api.dataset_status(manifest["results_dataset"])
    except Exception as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) != 404:
            raise
        response = api.dataset_create_new(str(archive_dir), public=False, quiet=True, convert_to_csv=False)
        assert_response_ok(response)
        return {"dataset": manifest["results_dataset"], "state": "created", "sha256": digest(archive)}
    return {"dataset": manifest["results_dataset"], "state": state, "sha256": digest(archive)}


def watch(api, manifest: dict, manifest_path: Path, interval: int, max_hours: float) -> None:
    started = time.monotonic()
    while (time.monotonic() - started) / 3600 < max_hours:
        try:
            current = status(api, manifest)
            manifest["last_status"] = current
            write_json(manifest_path, manifest)
            print(json.dumps(current), flush=True)
            if current["status"] in TERMINAL:
                files = download_outputs(api, manifest, manifest_path.parent / "output")
                manifest["downloaded_files"] = files
                try:
                    manifest["archive"] = archive_outputs(api, manifest, manifest_path.parent)
                except Exception as exc:
                    # Kernel outputs are already persistent on Kaggle and have
                    # been downloaded locally. A token lacking datasets.create
                    # must not turn a successful experiment into an endless
                    # watcher loop or trigger a second kernel submission.
                    manifest["archive"] = {
                        "state": "not_created",
                        "error_type": type(exc).__name__,
                        "local_output": str(manifest_path.parent / "output"),
                    }
                manifest["watcher_finished_utc"] = utc_now()
                write_json(manifest_path, manifest)
                print(json.dumps({"archived": manifest["archive"], "files": len(files)}), flush=True)
                return
        except Exception as exc:
            # Do not print request headers, full response objects, or secrets.
            manifest["watcher_last_error"] = {"type": type(exc).__name__, "at_utc": utc_now()}
            write_json(manifest_path, manifest)
            print(json.dumps(manifest["watcher_last_error"]), flush=True)
        time.sleep(interval)
    raise RuntimeError("Watcher time budget ended; artifacts remain retrievable from the private kernel.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-dir", type=Path, default=ROOT / "api_kaggle")
    parser.add_argument("--sdk-path", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--notebook", type=Path, required=True)
    prep.add_argument("--payload", type=Path, required=True)
    prep.add_argument("--label", required=True)
    prep.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    prep.add_argument(
        "--reuse-runtime-dataset",
        help="version this account's existing neuron-death-code Dataset",
    )
    for name in ("upload-runtime", "push", "status", "logs", "output", "watch"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        if name == "logs":
            sub.add_argument("--seconds", type=int, default=20)
        if name == "watch":
            sub.add_argument("--interval", type=int, default=90)
            sub.add_argument("--max-hours", type=float, default=18)
    args = parser.parse_args(argv)
    credentials = read_credentials(args.credentials_dir)
    if args.command == "prepare":
        result = prepare(
            args.notebook, args.payload, args.label, credentials["username"],
            args.accelerator, args.reuse_runtime_dataset,
        )
        print(result)
        return 0
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path, credentials["username"])
    api = api_client(args.credentials_dir, args.sdk_path)
    if args.command == "upload-runtime":
        print(upload_runtime(api, manifest, manifest_path))
    elif args.command == "push":
        print(json.dumps(push(api, manifest, manifest_path)))
    elif args.command == "status":
        print(json.dumps(status(api, manifest)))
    elif args.command == "logs":
        print(logs(api, manifest, manifest_path.parent / "logs", args.seconds)[-20000:])
    elif args.command == "output":
        print(json.dumps(download_outputs(api, manifest, manifest_path.parent / "output")))
    elif args.command == "watch":
        watch(api, manifest, manifest_path, max(30, args.interval), args.max_hours)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # No traceback locals, server body or credential values in console logs.
        print(f"Kaggle operation failed ({type(exc).__name__}).", file=sys.stderr)
        raise SystemExit(1)
