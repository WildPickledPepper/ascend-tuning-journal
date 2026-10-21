"""Audit original native-binary simulator schedules; never label cycles as board time.

Inputs: private raw popped/completed dumps, extracted CCU issue JSON, and Insight DBs.
Outputs: tensor-relative addresses, ELF-relative PCs, simulated cycle evidence only.
"""
import argparse
import collections
import hashlib
import json
import re
import sqlite3
import statistics
import struct
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("raw", type=Path)
p.add_argument("insight", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
LINE = re.compile(r"\[info\]\s+\[(\d+)\]\s+\(PC: (0x[\da-f]+)\)\s+(\w+)\s+:.*?\)\s+(\w+)\s*(.*)")
KEPT = {"MMAD", "MOV_OUT_TO_L1_MULTI_ND2NZ", "LOAD_2D", "FIX_L0C_TO_DST", "SET_FLAG", "WAIT_FLAG", "END"}
EXPECTED_ELF = "444d3ac5ab6c83b0321b32f7b2a66330de45b0eb215b0c44b8f62c3f2f2a227f"

def reg(detail, name):
    m = re.search(r"\b" + name + r":(?:X\d+=)?(0x[\da-f]+|0)\b", detail)
    assert m, (name, detail)
    return int(m[1], 0)

def parse_file(path):
    result = []
    for line in path.open(encoding="utf-8"):
        m = LINE.match(line)
        if m and m[4] in KEPT:
            tick, pc, pipe, name, detail = m.groups()
            event = dict(start=int(tick), pc=int(pc, 16), pipe=pipe, name=name, detail=detail)
            event["binary"] = int(re.search(r"\(Binary: (0x[\da-f]+)\)", line)[1],16)
            if name in {"SET_FLAG", "WAIT_FLAG"}:
                f = re.search(r"PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(\d+)", detail)
                event["flag"] = (f[1], f[2], int(f[3]))
            result.append(event)
    return result

def evidence_event(e, load_base):
    d = {k: e[k] for k in ["name", "start", "end", "issue"] if k in e}
    d["pc_offset"] = hex(e["pc"] - load_base)
    if "flag" in e:
        d["flag"] = list(e["flag"])
    return d

def analyze(shape, core):
    directory = a.raw / shape
    prefix = f"core{core}.cubecore0."
    pop_path = directory / (prefix + "instr_popped_log.dump")
    end_path = directory / (prefix + "instr_log.dump")
    events = parse_file(pop_path)
    ends = collections.defaultdict(list)
    for e in parse_file(end_path):
        ends[e["pc"]].append(e)
    counts = collections.Counter(e["pc"] for e in events)
    assert all(counts[pc] == len(es) for pc, es in ends.items())
    seen = collections.Counter()
    issue_data = json.loads((a.raw / f"{shape}-core{core}-issue.json").read_text())
    issues = collections.defaultdict(list)
    for e in issue_data["events"]:
        issues[e["pc"]].append(e)
    load_base = int((directory / "pc_start_addr.txt").read_text().strip(), 16)
    elf = (directory / "aicore_binary.o").read_bytes()
    header = struct.unpack_from("<16sHHIQQQIHHHHHH",elf)
    sections = [struct.unpack_from("<IIQQQQIIQQ",elf,header[6]+i*header[11]) for i in range(header[12])]
    strings = sections[header[13]]
    names = elf[strings[4]:strings[4]+strings[5]]
    text, = [s for s in sections if names[s[0]:].split(b"\0")[0] == b".text"]
    checked_pcs = set()
    for e in events:
        offset = e["pc"] - load_base
        assert struct.unpack_from("<I",elf,text[4]+offset-text[3])[0] == e["binary"]
        checked_pcs.add(offset)
        i = seen[e["pc"]]
        done = ends[e["pc"]][i]
        assert e["name"] == done["name"] and e["start"] <= done["start"]
        e["end"] = done["start"]
        if e["name"] not in {"SET_FLAG", "WAIT_FLAG", "END"}:
            assert len(issues[e["pc"]]) == counts[e["pc"]], (shape, core, e["name"])
            e["issue"] = issues[e["pc"]][i]["tick"]
            assert e["issue"] <= e["start"]
        seen[e["pc"]] += 1
    events.sort(key=lambda e: (e["start"], e["pc"]))
    assert sum(e["name"] == "END" for e in events) == 1
    sets, waits = collections.defaultdict(list), collections.defaultdict(list)
    for e in events:
        if e["name"] == "SET_FLAG": sets[e["flag"]].append(e)
        if e["name"] == "WAIT_FLAG": waits[e["flag"]].append(e)
    assert set(sets) == set(waits)
    for key in sets:
        assert len(sets[key]) == len(waits[key]), (shape, core, key)
        for s, w in zip(sets[key], waits[key]):
            assert s["end"] <= w["end"]
            w["producer_set"] = s
    copies, weights = [], []
    for e in events:
        if e["name"] == "MOV_OUT_TO_L1_MULTI_ND2NZ":
            xm = reg(e["detail"], "XM")
            e.update(rows=(xm >> 16) & 65535, columns=(xm >> 32) & 65535,
                     source=reg(e["detail"], "XN"), dest=reg(e["detail"], "XD"))
            e["l1_bytes"] = ((e["rows"] + 15) // 16 * 16) * e["columns"] * 2
            e["kind"] = "input" if e["rows"] == 8 else "weight"
            e["index"] = len(weights) if e["kind"] == "weight" else -1
            copies.append(e)
            if e["kind"] == "weight": weights.append(e)
    base_events = parse_file(directory / "core0.cubecore0.instr_popped_log.dump")
    weight_base = min(reg(e["detail"],"XN") for e in base_events
                      if e["name"] == "MOV_OUT_TO_L1_MULTI_ND2NZ" and (reg(e["detail"],"XM")>>16)&65535 != 8)
    k = 5120 if shape == "up" else 17408
    for e in weights:
        e["N"], byte_k = divmod(e["source"] - weight_base, k * 2)
        e["K"] = byte_k // 2
    loads = [e for e in events if e["name"] == "LOAD_2D" and "Dst:L0B" in e["detail"]]
    for e in loads:
        src = reg(e["detail"], "XN")
        candidates = [w for w in weights if w["dest"] <= src < w["dest"] + w["l1_bytes"] and w["end"] <= e["start"]]
        assert candidates
        e["weight"] = max(candidates, key=lambda x: x["start"])
        e["l0_dest"] = reg(e["detail"], "XD")
    mmads = [e for e in events if e["name"] == "MMAD"]
    by_weight = collections.defaultdict(list)
    for e in mmads:
        operand = reg(e["detail"], "XM")
        candidates = [l for l in loads if l["l0_dest"] == operand and l["end"] <= e["start"]]
        assert candidates
        l = max(candidates, key=lambda x: x["start"])
        e["weight"] = l["weight"]
        by_weight[l["weight"]["index"]].append(e)
    expected_chunks = 4 if shape == "up" else 8
    assert len(mmads) == len(weights) * expected_chunks
    assert all(len(by_weight[w["index"]]) == expected_chunks for w in weights)
    for w in weights:
        w["mmads"] = by_weight[w["index"]]
        w["loads"] = [l for l in loads if l["weight"] is w]
        assert len(w["loads"]) == expected_chunks
    fixps = [e for e in events if e["name"] == "FIX_L0C_TO_DST"]
    boundaries = []
    inputs = [e for e in copies if e["kind"] == "input"]
    input_base = min(e["source"] for e in inputs)
    for prev, nxt in zip(weights, weights[1:]):
        if prev["N"] == nxt["N"]: continue
        last_mmad = max(prev["mmads"], key=lambda e: e["end"])
        last_load = max(prev["loads"], key=lambda e: e["end"])
        key = ("MTE1", "MTE2", (0 if nxt["dest"] == 0x28000 else 1) if shape == "up" else (2 if nxt["dest"] == 0x8000 else 3))
        prior_waits = [w for w in waits[key] if w["start"] <= nxt["start"]]
        release_wait = max(prior_waits, key=lambda w: w["start"])
        release_set = release_wait["producer_set"]
        old = [w for w in weights if w["index"] < nxt["index"] and w["dest"] == nxt["dest"]]
        old = old[-1]
        assert max(l["end"] for l in old["loads"]) <= release_set["end"] <= nxt["issue"]
        ready_key = ("MTE2", "MTE1", 0 if shape == "up" else 1)
        ready = min((w for w in waits[ready_key] if w["end"] >= nxt["end"]), key=lambda w:w["end"])
        assert ready["producer_set"]["end"] == nxt["end"]
        first_mmad = min(nxt["mmads"],key=lambda e:e["start"])
        consumer_wait = max((w for w in waits[("MTE1","CUBE",0)] if w["end"] <= first_mmad["start"]),key=lambda e:e["end"])
        first_load = min(nxt["loads"],key=lambda e:e["start"])
        assert first_load["end"] <= consumer_wait["producer_set"]["end"] <= consumer_wait["end"]
        next_input = inputs[nxt["index"]] if shape == "down" else inputs[0]
        record = {
            "old_N": prev["N"], "next_N": nxt["N"], "next_first_K": nxt["K"],
            "next_weight_copy_index": nxt["index"], "l1_slot_offset_bytes": nxt["dest"],
            "copy_shape_NK": [nxt["rows"], nxt["columns"]],
            "next_weight_source_offset_bytes": nxt["source"] - weight_base,
            "source_weight_row_pitch_bytes": k * 2,
            "prior_weight_copy": evidence_event(prev,load_base),
            "prior_last_mmad": evidence_event(last_mmad,load_base),
            "prior_last_L1_to_L0B": evidence_event(last_load,load_base),
            "next_weight_copy": evidence_event(nxt,load_base),
            "next_input_copy": evidence_event(next_input,load_base),
            "next_input_source_offset_bytes": next_input["source"] - input_base,
            "next_input_slot_offset_bytes": next_input["dest"],
            "prior_result_write": evidence_event(fixps[len(boundaries)],load_base),
            "next_first_L1_to_L0B": evidence_event(first_load,load_base),
            "next_first_mmad": evidence_event(nxt["mmads"][0],load_base),
            "slot_release_set": evidence_event(release_set,load_base),
            "slot_reuse_wait": evidence_event(release_wait,load_base),
            "next_data_ready_wait": evidence_event(ready,load_base),
            "next_cube_ready_set": evidence_event(consumer_wait["producer_set"],load_base),
            "next_cube_ready_wait": evidence_event(consumer_wait,load_base),
            "copy_issue_before_old_last_mmad_end_cycles": last_mmad["end"] - nxt["issue"],
            "copy_start_before_old_last_mmad_end_cycles": last_mmad["end"] - nxt["start"],
            "safe_slot_available_before_actual_issue_cycles": nxt["issue"] - release_set["end"],
            "slot_and_prior_weight_transfer_done_before_actual_issue_cycles": nxt["issue"] - max(release_set["end"],prev["end"]),
            "old_mmads_not_finished_when_next_copy_issued": sum(m["end"] > nxt["issue"] for m in mmads if m["weight"]["N"] == prev["N"]),
        }
        if shape == "down":
            input_flag = 0 if next_input["dest"] == 0 else 1
            input_wait = max((w for w in waits[("MTE1","MTE2",input_flag)] if w["start"] <= next_input["start"]),key=lambda e:e["start"])
            input_set = input_wait["producer_set"]
            record["next_input_slot_release_set"] = evidence_event(input_set,load_base)
            record["next_input_slot_reuse_wait"] = evidence_event(input_wait,load_base)
            record["both_slots_and_prior_weight_transfer_done_before_actual_weight_issue_cycles"] = nxt["issue"] - max(input_set["end"],release_set["end"],prev["end"])
        boundaries.append(record)
    transitions = []
    for prev, nxt in zip(weights, weights[1:]):
        last = max(prev["mmads"],key=lambda e:e["end"])
        transitions.append(dict(cross_N=prev["N"] != nxt["N"], lead_cycles=last["end"]-nxt["issue"]))
    # Validate that raw simulated start/end agree with the compiled Insight import.
    db = a.insight / (shape + "-visualize_data_mindstudio_insight_data.db")
    with sqlite3.connect("file:"+db.as_posix()+"?mode=ro",uri=True) as c:
        for name in ["MMAD", "MOV_OUT_TO_L1_MULTI_ND2NZ", "LOAD_2D", "FIX_L0C_TO_DST"]:
            imported = list(c.execute("select s.timestamp,s.duration,s.args from slice s join thread t using(track_id) where t.pid=? and s.name=? order by s.timestamp", (f"core{core}.cubecore0",name)))
            raw = sorted((e for e in events if e["name"] == name),key=lambda e:e["start"])
            assert len(imported) == len(raw), (shape,core,name)
            for (ts,duration,args),e in zip(imported,raw):
                assert int(json.loads(args)["pc_addr"],16) == e["pc"]
                assert abs(ts-e["start"]/1.8) <= 1.1
                assert abs(ts+duration-e["end"]/1.8) <= 1.1
    return {
        "core": core, "weight_copies":len(weights), "MMAD_count":len(mmads), "output_tiles":len(fixps),
        "all_raw_transfer_and_MMAD_times_match_Insight":True,
        "all_kept_opcodes_match_original_ELF":True,
        "unique_opcodes_checked":len(checked_pcs),
        "popped_sha256":hashlib.sha256(pop_path.read_bytes()).hexdigest(),
        "completed_sha256":hashlib.sha256(end_path.read_bytes()).hexdigest(),
        "ccu_sources_sha256":issue_data["hashes"],
        "cross_output_boundaries":boundaries,
        "within_K_transitions":{
            "total":sum(not t["cross_N"] for t in transitions),
            "next_copy_issued_before_prior_last_mmad_finishes":sum(not t["cross_N"] and t["lead_cycles"]>0 for t in transitions),
            "copy_issue_lead_cycles_min_median_max": [min(t["lead_cycles"] for t in transitions if not t["cross_N"]),statistics.median(t["lead_cycles"] for t in transitions if not t["cross_N"]),max(t["lead_cycles"] for t in transitions if not t["cross_N"])],
        },
    }

result = {"kind":"native ELF + simulator CCU issue / popped / completed + compiled Insight DB",
          "timing_unit":"SIMULATED cycles, NOT measured board time", "elf_sha256":EXPECTED_ELF,"operators":{},
          "limitations":[
              "Original same-shape F.linear probe: not a new capture of every model-layer launch.",
              "Original blockDim=23; only cores 0, 1 and 22 dynamically inspected here.",
              "CCU issue and simulator execution/completion are separate timestamps, all simulated.",
              "No measured device WAIT latency, modified-kernel benchmark or new model throughput result.",
              "Free L1 slots establish a scheduling opportunity, not a measured achievable speedup.",
              "Direct llvm-objdump instruction text was unavailable; raw simulator opcode words were checked against ELF .text instead.",
          ]}
for shape in ["up","down"]:
    assert hashlib.sha256((a.raw/shape/"aicore_binary.o").read_bytes()).hexdigest() == EXPECTED_ELF
    result["operators"][shape] = [analyze(shape,core) for core in [0,1,22]]
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
for shape,cores in result["operators"].items():
    for c in cores:
        print(shape,c["core"],c["within_K_transitions"],[(b["old_N"],b["next_N"],b["copy_issue_before_old_last_mmad_end_cycles"]) for b in c["cross_output_boundaries"]])
