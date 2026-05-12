from pathlib import Path

from pipeline_runner import SyllabusPipelineConfig, build_train_argv, find_repo_root


def test_find_repo_root_from_cwd(tmp_path: Path):
    repo = tmp_path / "training_pipeline"
    repo.mkdir()
    (repo / "build_finetune_dataset.py").write_text("# stub\n", encoding="utf-8")
    found = find_repo_root(repo)
    assert found.resolve() == repo.resolve()


def test_build_train_argv_includes_paths_and_bf16(tmp_path: Path):
    repo = tmp_path / "training_pipeline"
    repo.mkdir()
    (repo / "build_finetune_dataset.py").write_text("# stub\n", encoding="utf-8")
    cfg = SyllabusPipelineConfig.for_repo(repo)
    cfg.train_jsonl = repo / "data" / "finetune" / "train.jsonl"
    cfg.valid_jsonl = repo / "data" / "finetune" / "valid.jsonl"
    cfg.model_output_dir = repo / "artifacts" / "out"
    cfg.bf16 = True
    cfg.disable_mlflow = True
    argv = build_train_argv(cfg)
    assert argv[0] == "train_hf_structured_extractor.py"
    assert "--train-jsonl" in argv
    assert str(cfg.train_jsonl) in argv
    assert "--bf16" in argv
    assert "--disable-mlflow" in argv
