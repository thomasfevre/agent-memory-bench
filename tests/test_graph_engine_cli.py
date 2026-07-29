from graph_engine_cli import parse_cognee_args


def test_cognee_cli_exposes_local_provider_concurrency_controls(tmp_path):
    args = parse_cognee_args(
        [
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--questions",
            str(tmp_path / "questions.jsonl"),
            "--output",
            str(tmp_path / "result.json"),
            "--chunks-per-batch",
            "1",
            "--data-per-batch",
            "1",
        ]
    )

    assert args.chunks_per_batch == 1
    assert args.data_per_batch == 1
