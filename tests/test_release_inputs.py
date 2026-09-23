from nema.data import pair_paths


def test_native_sanitation_does_not_change_retrieval_or_training(tmp_path):
    for relative in ("MCES/MOLHIV-test/raw", "MCES/MOLHIV-train/raw", "retrieval/MOLHIV/raw/test"):
        folder = tmp_path / relative
        folder.mkdir(parents=True)
        for number in (1, 2, 23, 46, 48, 54, 61, 64, 76):
            (folder / f"graphs_{number}.pkl").touch()
    assert [p.stem for p in pair_paths(tmp_path, "MOLHIV")] == ["graphs_1", "graphs_2"]
    assert len(pair_paths(tmp_path, "MOLHIV", split="train")) == 9
    assert len(pair_paths(tmp_path, "MOLHIV", retrieval=True)) == 9
    assert [p.stem for p in pair_paths(tmp_path, "MOLHIV", limit=1)] == ["graphs_1"]
