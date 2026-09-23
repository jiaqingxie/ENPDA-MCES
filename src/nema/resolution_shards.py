"""Auditable deterministic sharding for long-timeout RASCAL resolutions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys_sha256(keys: list[int]) -> str:
    payload = json.dumps(keys, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl_unique(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = int(record["key"])
        if key in records:
            raise ValueError(f"duplicate key {key} in {path}")
        records[key] = record
    return records


def build_plan(
    oracle_path: Path,
    classification_path: Path,
    shard_count: int,
    timeout_seconds: int,
) -> dict:
    if shard_count < 1 or timeout_seconds < 1:
        raise ValueError("shard count and timeout must be positive")
    oracle_sha256 = sha256_path(oracle_path)
    classification_sha256 = sha256_path(classification_path)
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    if classification.get("oracle_sha256") != oracle_sha256:
        raise ValueError("classification was not generated from the current oracle")
    oracle = read_jsonl_unique(oracle_path)
    required_keys = sorted(
        int(record["key"])
        for record in classification.get("records", [])
        if bool(record.get("official_mrr_p10_map_resolution_required", True))
    )
    if len(required_keys) != len(set(required_keys)):
        raise ValueError("classification contains duplicate required keys")
    for key in required_keys:
        if key not in oracle or not bool(oracle[key].get("rascal_timed_out", False)):
            raise ValueError(f"required key is not a current oracle timeout: {key}")
    shards = []
    for index in range(shard_count):
        keys = required_keys[index::shard_count]
        shards.append(
            {
                "index": index,
                "key_count": len(keys),
                "keys_sha256": keys_sha256(keys),
                "keys": keys,
            }
        )
    first = oracle[min(oracle)]
    return {
        "schema_version": 1,
        "protocol": "deterministic-round-robin-required-key-sharding-v1",
        "dataset": Path(oracle_path).parent.name,
        "seed": int(first["protocol_seed"]),
        "oracle_path": str(oracle_path),
        "oracle_sha256": oracle_sha256,
        "classification_path": str(classification_path),
        "classification_sha256": classification_sha256,
        "relevance_threshold": float(classification["relevance_threshold"]),
        "resolution_timeout_seconds": timeout_seconds,
        "shard_count": shard_count,
        "required_key_count": len(required_keys),
        "required_keys_sha256": keys_sha256(required_keys),
        "required_keys": required_keys,
        "assignment": "sorted required keys, round-robin by zero-based position modulo shard_count",
        "shards": shards,
    }


def verify_plan(path: Path, *, verify_inputs: bool = True) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if int(plan.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported resolution-shard plan schema: {path}")
    shard_count = int(plan["shard_count"])
    shards = plan["shards"]
    if shard_count < 1 or len(shards) != shard_count:
        raise ValueError("resolution-shard plan has the wrong shard count")
    required = [int(key) for key in plan["required_keys"]]
    if required != sorted(required) or len(required) != len(set(required)):
        raise ValueError("plan required keys must be sorted and unique")
    if len(required) != int(plan["required_key_count"]):
        raise ValueError("plan required-key count mismatch")
    if keys_sha256(required) != plan["required_keys_sha256"]:
        raise ValueError("plan required-key hash mismatch")
    flattened: list[int] = []
    for expected_index, shard in enumerate(shards):
        if int(shard["index"]) != expected_index:
            raise ValueError("plan shard indices are not contiguous")
        keys = [int(key) for key in shard["keys"]]
        if keys != required[expected_index::shard_count]:
            raise ValueError(f"plan shard {expected_index} is not deterministic round-robin")
        if len(keys) != int(shard["key_count"]) or keys_sha256(keys) != shard["keys_sha256"]:
            raise ValueError(f"plan shard {expected_index} key audit mismatch")
        flattened.extend(keys)
    if len(flattened) != len(set(flattened)) or sorted(flattened) != required:
        raise ValueError("plan shards are overlapping or do not cover every required key")
    if verify_inputs:
        oracle_path = Path(plan["oracle_path"])
        classification_path = Path(plan["classification_path"])
        if sha256_path(oracle_path) != plan["oracle_sha256"]:
            raise ValueError("current oracle hash differs from frozen shard plan")
        if sha256_path(classification_path) != plan["classification_sha256"]:
            raise ValueError("current classification hash differs from frozen shard plan")
        classification = json.loads(classification_path.read_text(encoding="utf-8"))
        if classification.get("oracle_sha256") != plan["oracle_sha256"]:
            raise ValueError("classification/oracle hash mismatch in shard plan")
        classified_required = sorted(
            int(record["key"])
            for record in classification.get("records", [])
            if bool(record.get("official_mrr_p10_map_resolution_required", True))
        )
        if classified_required != required:
            raise ValueError("classification required keys differ from frozen shard plan")
    return plan


def shard_keys(plan: dict, shard_index: int) -> list[int]:
    if not 0 <= shard_index < int(plan["shard_count"]):
        raise ValueError(f"shard index out of range: {shard_index}")
    return [int(key) for key in plan["shards"][shard_index]["keys"]]


def shard_directory(plan_path: Path, shard_index: int) -> Path:
    plan = verify_plan(plan_path, verify_inputs=False)
    return plan_path.parent / f"shard-{shard_index:02d}-of-{int(plan['shard_count']):02d}"


def validate_gpu_provenance(path: Path) -> dict:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if "H100" not in str(provenance.get("device", "")):
        raise ValueError("resolution shard refused non-H100 provenance")
    if provenance.get("cuda_runtime") != "12.8":
        raise ValueError("resolution shard refused non-CUDA-12.8 provenance")
    for raw_path, expected_sha256 in provenance.get("code_sha256", {}).items():
        code_path = Path(raw_path)
        if not code_path.is_file() or sha256_path(code_path) != expected_sha256:
            raise ValueError(f"resolution shard code provenance mismatch: {code_path}")
    return provenance


def validate_shard_artifacts(
    plan_path: Path,
    shard_index: int,
    resolution_path: Path,
    provenance_path: Path,
) -> dict:
    plan = verify_plan(plan_path)
    assigned = shard_keys(plan, shard_index)
    oracle = read_jsonl_unique(Path(plan["oracle_path"]))
    resolutions = read_jsonl_unique(resolution_path)
    if set(resolutions) != set(assigned):
        missing = sorted(set(assigned) - set(resolutions))
        extra = sorted(set(resolutions) - set(assigned))
        raise ValueError(
            f"resolution shard coverage mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    minimum_timeout = int(plan["resolution_timeout_seconds"])
    timed_out = 0
    for key, record in resolutions.items():
        source = oracle[key]
        if str(record["source_path"]) != str(source["source_path"]):
            raise ValueError(f"resolution source mismatch for key {key}")
        if abs(float(record["previous_similarity"]) - float(source["true_similarity"])) > 1e-12:
            raise ValueError(f"resolution previous similarity mismatch for key {key}")
        if int(record["previous_common_edges"]) != int(source["true_edges"]):
            raise ValueError(f"resolution previous edge truth mismatch for key {key}")
        if int(record["previous_common_nodes"]) != int(source["true_nodes"]):
            raise ValueError(f"resolution previous node truth mismatch for key {key}")
        if int(record["rascal_timeout_seconds"]) < minimum_timeout:
            raise ValueError(f"resolution timeout below frozen plan for key {key}")
        timed_out += bool(record["rascal_timed_out"])
    provenance = validate_gpu_provenance(provenance_path)
    return {
        "plan": plan,
        "assigned_keys": assigned,
        "resolutions": resolutions,
        "provenance": provenance,
        "timed_out_pairs": timed_out,
    }
