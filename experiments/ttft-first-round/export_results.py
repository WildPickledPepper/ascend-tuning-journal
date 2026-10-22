"""Verify bounded natural-output runs and export path-free evidence."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "count-ready-overlap"))
from latency_statistics import summarize_rows
from summarize import acceptance


def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_check(a):
    assert a["dtype"] == "torch.bfloat16" and a["tp_size"] == 1 and a["num_speculative_tokens"] == 7
    assert a["draft_use_cuda_graph"] and not a["draft_enforce_eager"]
    assert a["gdn_state"]["layer_count"] == 48 and a["gdn_state"]["all_ssm_fp32"] and a["gdn_state"]["all_conv_bf16"]
    assert a["layout_experiment"]["packed_count"] == 64 and a["layout_experiment"]["prefill_direct_enabled"]
    assert a["dynamic_vocabulary"]["capacity"] == 4096
    assert all(any(e["captured"] for e in w["entries"]) for w in a["wrappers"])


def arm(root, label):
    r = root / "results"
    rows = load(r / (label + ".json"))
    assert len(rows) == 32 and all(x["finish_reason"] == "stop" and len(x["token_ids"]) < 2048 for x in rows)
    result = summarize_rows(rows, load(r / (label + "-window.json")))
    result["acceptance"] = acceptance(r, label)
    result["per_request"] = [{k: x[k] for k in ("question_id", "e2e_s", "ttft_s", "tpot_ms", "finish_reason")} |
                             {"input_tokens": x["usage"]["prompt_tokens"], "output_tokens": len(x["token_ids"]),
                              "sequence_sha256": hashlib.sha256(json.dumps(x["token_ids"]).encode()).hexdigest()} for x in rows]
    a, b = [load(r / (label + s + "-audit.json")) for s in ("-before", "-after")]
    model_check(a); model_check(b)
    va, vb = a["dynamic_vocabulary"], b["dynamic_vocabulary"]
    assert (va["ids_pointer"], va["packed_pointer"]) == (vb["ids_pointer"], vb["packed_pointer"])
    count_a, count_b = a["layout_experiment"]["counts"], b["layout_experiment"]["counts"]
    assert count_b.get("gate_unpack_fallback_host_calls", 0) == count_a.get("gate_unpack_fallback_host_calls", 0)
    overlap = load(r / (label + "-overlap-audit.json"))["counts"]
    assert len({overlap[k] for k in ("sample_calls", "early_vocab_updates", "valid_count_primed", "valid_count_prepare_reused", "valid_count_copy_reused")}) == 1
    copy_audit = r / (label + "-first-copy-audit.json")
    if copy_audit.exists():
        c = load(copy_audit)
        assert c["pending"] == 0 and c["source"]["constructor_fields_checked"] == 11
        if label == "optimized32":
            assert c["enabled"] and c["counts"]["first_copies_primed"] == c["counts"]["first_copies_consumed"] == 33
            assert c["source"]["core_changed_clauses"] == 2 and c["counts"]["prefill_outputs_flushed"] == 33
        result["first_copy_audit"] = c
    result["audit"] = {"model_graph_dtype_and_layout_valid": True, "vocabulary_buffer_addresses_stable": True,
                       "prefill_unpack_fallback_delta": 0, "counters": overlap}
    return result, rows


def diagnosis(root):
    if not (root / "results/diagnosis.json").exists():
        return []
    raw = load(root / "results/diagnosis.json")
    output = []
    for t in raw:
        scopes = [{"name": e["name"], "wall_ms": (e["end_ns"] - e["start_ns"]) / 1e6,
                   "cpu_ms": e["cpu_ms"]} for e in t["audit"]["events"]]
        stack, pauses = {}, []
        for e in t["audit"]["gc_events"]:
            if e["phase"] == "start":
                stack[e["generation"]] = e["wall_ns"]
            elif e["generation"] in stack:
                pauses.append((e["wall_ns"] - stack.pop(e["generation"])) / 1e6)
        output.append({"repeat": t["repeat"], "question_id": t["request"]["question_id"], "mode": t["mode"],
                       "ttft_ms": t["request"]["ttft_s"] * 1000,
                       "execute_device_event_ms": t["audit"]["execute_device_event_ms"],
                       "scopes": scopes, "max_gc_pause_ms": max(pauses, default=0)})
    return output


def main(r1, r2, out):
    a, raw_a = arm(r2, "control32")
    b, raw_b = arm(r2, "optimized32")
    earlier, raw_earlier = arm(r1, "queue1-ready3")
    assert all(x["token_ids"] == y["token_ids"] and x["usage"]["prompt_tokens"] == y["usage"]["prompt_tokens"] for x, y in zip(raw_a, raw_b))
    assert all(x["question_id"] == y["question_id"] == z["question_id"] and
               x["token_ids"] == y["token_ids"] == z["token_ids"]
               for x, y, z in zip(raw_a, raw_b, raw_earlier))
    checks = []
    v = load(r2 / "results/validation-early_output.json")
    for i in range(0, len(v), 3):
        first, opt, again = v[i:i + 3]
        assert first["request"]["token_ids"] == opt["request"]["token_ids"] == again["request"]["token_ids"]
        check = {"question_id": first["question_id"], "outputs_equal": True}
        for k in ("candidate_set_sequence_sha256", "sampled_sequence_sha256", "draft_sequence_sha256"):
            repeated = first["audit"][k] == again["audit"][k]
            if repeated:
                assert first["audit"][k] == opt["audit"][k]
            check[k.replace("_sequence_sha256", "")] = {
                "baseline_repeats": repeated,
                "optimized_equal": first["audit"][k] == opt["audit"][k],
                "optimized_equal_second_baseline": opt["audit"][k] == again["audit"][k],
                "recorded_round_counts": [len(x["audit"][k]) for x in (first, opt, again)]}
        checks.append(check)
    assert len(checks) == 2
    result = {"model": "Qwen3.8-27B", "physical_card": 7, "earlier_diagnosis_physical_card": 6, "TP": 1, "concurrency": 1,
              "weights": "BF16", "ssm_state": "FP32", "MTP": 7, "same_round_dynamic_vocabulary": 4096,
              "control": a, "optimized": b, "earlier_reference": earlier, "validation": checks,
              "comparison": {"all32_outputs_and_prompt_lengths_equal": True,
                  "all32_outputs_equal_earlier_reference": True,
                  "throughput_change_pct": 100 * (b["common"]["output_throughput_by_request_e2e"] / a["common"]["output_throughput_by_request_e2e"] - 1),
                  "window_throughput_change_pct": 100 * (b["common"]["output_throughput_by_window"] / a["common"]["output_throughput_by_window"] - 1),
                  "ttft_mean_change_pct": 100 * (b["distributions"]["TTFT"]["mean"] / a["distributions"]["TTFT"]["mean"] - 1),
                  "faster_e2e_requests": sum(y["e2e_s"] < x["e2e_s"] for x, y in zip(raw_a, raw_b)),
                  "lower_ttft_requests": sum(y["ttft_s"] < x["ttft_s"] for x, y in zip(raw_a, raw_b))},
              "diagnosis_queue1": diagnosis(r1), "diagnosis_candidate_control": diagnosis(r2),
              "limitations": ["Final comparison is one sequential 32-question arm each in the SAME service, original host allocator, not randomized repeated trials. Earlier-service control is context only.",
                  "One formal 32-question run per configuration; does not guarantee every request or other workloads improve.",
                  "The historical 499ms request has no detailed first-round trace; its exact cause cannot be reconstructed.",
                  "CPU scope wall minus thread CPU includes waits and descheduling; device Event intervals include submission gaps.",
                  "Telemetry, warmup and candidate diagnostics are disabled/excluded from the formal benchmark.",
                  "Preflight uses original stage0 -> optimized stage3+delivery -> original stage0. Both original repeats have differing full candidate/sample/draft sequences; the conditional trajectory equality gate is inconclusive, not a passed full-trajectory equivalence test. Final control is stage3, not stage0.",
                  "Formal 32-request acceptance rounds are 1870 vs 1872. Outputs match, but complete speculative trajectories were not recorded and are not proven equal.",
                  "MTP streaming event intervals are not device-internal single-token ITL."]}
    result["private_artifact_hashes"] = {}
    short_rows, short = {}, {}
    for label in ("same-service-control4", "early-copy4"):
        data = load(r2 / "results" / (label + ".json"))
        assert len(data) == 4
        short_rows[label] = data
        n, elapsed = sum(len(x["token_ids"]) for x in data), sum(x["e2e_s"] for x in data)
        short[label] = {"output_tokens": n, "e2e_sum_s": elapsed, "throughput": n / elapsed,
                        "ttft_mean_ms": sum(x["ttft_s"] for x in data) * 250,
                        "first_copy": load(r2 / "results" / (label + "-first-copy-audit.json"))}
    assert all(a["token_ids"] == b["token_ids"] for a, b in zip(*short_rows.values()))
    result["same_service_short_comparison"] = short
    result["rejected_trials"] = load(r2 / "rejected-trials.json")
    units = load(r2 / "results/unit-cleanup.json")
    assert units["own_units_inactive"] and not units["other_units_stopped"]
    result["own_unit_cleanup"] = units
    for tag, root, label in (("earlier_reference", r1, "queue1-ready3"), ("control", r2, "control32"), ("optimized", r2, "optimized32")):
        result["private_artifact_hashes"][tag] = {p.name: sha(p) for p in (root / "results").iterdir()
                                                if p.is_file() and (p.name.startswith(label) or p.name in ("diagnosis.json", "validation-early_output.json", "cleanup.json"))}
        cleanup = load(root / "results/cleanup.json")
        assert cleanup["own_service_stopped"] and cleanup["recorded_own_process_ids_absent"] and cleanup["port8047_closed"]
        result.setdefault("cleanup", {})[tag] = cleanup
    manifests = [load(root / "archive-manifest.json") for root in (r1, r2)]
    for root, manifest in zip((r1, r2), manifests):
        assert sha(root / manifest["archive"]) == manifest["sha256"]
    def benchmark_body(root, manifest):
        with tarfile.open(root / manifest["archive"]) as t:
            tree = ast.parse(t.extractfile("driver.py").read().decode())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                    and n.test.left.id == "action" and any(isinstance(c, ast.Constant) and c.value == "benchmark" for c in n.test.comparators))
        loop = next(n for n in node.body if isinstance(n, ast.For))
        return ast.dump(loop, include_attributes=False)
    assert benchmark_body(r1, manifests[0]) == benchmark_body(r2, manifests[1])
    before = load(r2 / "runtime-provenance-before.json")
    after = load(r2 / "runtime-provenance-after.json")
    assert before == after
    assert all(not after[name]["tracked_diff"] for name in ("vllm", "vllm-ascend"))
    options = load(r2 / "results/worker-options.json")
    assert options["own_worker_options"]["TASK_QUEUE_ENABLE"] == "1"
    assert options["physical_card"] == 7 and options["own_worker_pid_matches_card_probe"]
    assert not options["jemalloc_in_own_worker_maps"]
    assert options["own_worker_options"]["ASCEND_LAUNCH_BLOCKING"] != "1"
    result["provenance"] = {"archives": manifests, "runtime_sources_midrun_and_after_equal": True,
                            "source_check_timing": "Source/library disk hashes checked during final optimized arm and after stop; both arms use the same already-loaded model, graphs and libraries, also audited before/after each arm.",
                            "runtime": after, "candidate_worker_options": options,
                            "client_measurement_loop_ast_equal": True,
                            "local_analysis_sha256": sha(Path(__file__))}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
