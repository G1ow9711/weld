"""Bounded local inventory for the new small-sample route; no model updates."""
import json
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776"
def read(p): return json.loads((ROOT / p).read_text("utf-8-sig"))
def main():
    OUT.mkdir(exist_ok=True)
    protocol = read("weld_pipeline/reports/goal_contract_training_20261005/owned_emission_free65_466_attempt1/registered_protocol.json")
    manifest = read(protocol["scoring_manifest_path"])
    book = read("weld_pipeline/reports/full65_continuous_material_341_20261005/full65_source_book_341.json")
    raw = protocol["original_PCDFiles"]
    rows = []
    for i, (entry, source) in enumerate(zip(manifest, book["sources"])):
        target = read(entry["target_path"])
        children = [c for p in target["parents"] for c in p["children"]]
        types = Counter(c["weld_type"] + "/" + c["original_kind"] for c in children)
        gaps = Counter(str(g) for p in target["parents"] for g in p["gaps"])
        frames = source["verified_offline_frames"]
        cad = read(frames[0]["catalogue_path"]) if frames else {}
        row = dict(index=i, name=Path(source.get("STEP_path") or raw[i]["original_pcd_path"]).stem,
                   points=entry["original_points"], G=len(children), categories=dict(types),
                   max_children=max(len(p["children"]) for p in target["parents"]),
                   gaps=dict(gaps), frames=len(frames), volumes=len(cad.get("volumes", [])),
                   raw=raw[i], target=entry, step=source)
        rows.append(row)
        print(i, row["name"], row["points"], row["G"], dict(types), "multi", row["max_children"],
              "frames",len(frames),"vol",row["volumes"],"gaps",str(dict(gaps))[:110],flush=True)
    (OUT/"inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    first=rows[0]
    target=read(first["target"]["target_path"])
    cad=read(first["step"]["verified_offline_frames"][0]["catalogue_path"])
    print("TARGET_EXAMPLE", json.dumps(target["parents"][0],ensure_ascii=False)[:4500])
    print("CAD_EXAMPLE", json.dumps(cad["volumes"][0],ensure_ascii=False)[:2000])
    print("FRAME_EXAMPLE", json.dumps(first["step"]["verified_offline_frames"][0],ensure_ascii=False))
if __name__ == "__main__": main()
