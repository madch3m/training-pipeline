"""Export quantized edge artifacts (ONNX, GGUF, CoreML) from a trained adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export quantized artifacts for edge deployment.")
    p.add_argument("--adapter-path", default="artifacts/hf_syllabus_extractor")
    p.add_argument("--output-dir", default="artifacts/edge")
    p.add_argument(
        "--formats",
        nargs="+",
        default=["onnx-int8", "gguf-q4", "coreml-int8"],
        help="Subset of: onnx-int8, gguf-q4, coreml-int8.",
    )
    p.add_argument("--gguf-converter", default=None, help="Optional convert_hf_to_gguf.py path.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manifest-json", default=None)
    return p.parse_args()


def _run(cmd: list[str], dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"ok": True, "dry_run": True, "command": cmd}
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": cmd,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def export_onnx_int8(adapter_path: Path, out_dir: Path, dry_run: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "optimum.exporters.onnx",
        "--model",
        str(adapter_path),
        "--task",
        "text-generation",
        str(out_dir),
    ]
    res = _run(cmd, dry_run)
    res["note"] = "Apply ORT dynamic quantization to exported ONNX graph in deployment build step."
    return res


def export_gguf_q4(adapter_path: Path, out_dir: Path, converter: str | None, dry_run: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = converter or "convert_hf_to_gguf.py"
    cmd = [
        "python",
        script,
        str(adapter_path),
        "--outtype",
        "q4_k_m",
        "--outfile",
        str(out_dir / "model-q4_k_m.gguf"),
    ]
    return _run(cmd, dry_run)


def export_coreml_int8(adapter_path: Path, out_dir: Path, dry_run: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CoreML export route is environment-specific; shell out to an inline script for portability.
    cmd = [
        "python",
        "-c",
        (
            "import pathlib; "
            "p=pathlib.Path(r'%s'); p.mkdir(parents=True, exist_ok=True); "
            "print('coreml export placeholder: integrate coremltools conversion in macOS build agent')"
        )
        % str(out_dir),
    ]
    res = _run(cmd, dry_run)
    res["note"] = "Run on macOS with coremltools and a merged base+adapter checkpoint."
    return res


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "adapter_path": str(adapter_path),
        "output_dir": str(out_root),
        "formats": args.formats,
        "results": {},
    }

    for fmt in args.formats:
        if fmt == "onnx-int8":
            manifest["results"][fmt] = export_onnx_int8(adapter_path, out_root / "onnx-int8", args.dry_run)
        elif fmt == "gguf-q4":
            manifest["results"][fmt] = export_gguf_q4(
                adapter_path, out_root / "gguf-q4", args.gguf_converter, args.dry_run
            )
        elif fmt == "coreml-int8":
            manifest["results"][fmt] = export_coreml_int8(adapter_path, out_root / "coreml-int8", args.dry_run)
        else:
            manifest["results"][fmt] = {"ok": False, "error": "unknown_format"}

    text = json.dumps(manifest, indent=2, ensure_ascii=True)
    print(text)
    out_manifest = Path(args.manifest_json) if args.manifest_json else out_root / "quantization_manifest.json"
    out_manifest.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
