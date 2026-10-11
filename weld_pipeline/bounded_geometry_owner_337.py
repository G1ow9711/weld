"""Original-Popen geometry job ownership; all registered resource gates apply."""
import os
import shutil
import time
from .live_json_snapshot_282 import write_json_snapshot as dump
from .world_set_training_285 import utc
from .reports.serial_training_recovery_310_20261004.verify_first_actual_resume_update_310 import read, digest
from .reports.engineering_resume_274_20261003.owned_native_watch_274 import NATIVE, launch_owned
from .reports.registered_local_aux_resume_299_20261004.owned_resume_watch_299 import identity, watch
from .reports.full25_source_geometry_263_20261003.run_fresh_source_geometry_263 import process_identity, memory


def execute_owned_geometry(root, owner, output, script, config, validate_result):
    dump(owner / "controller_identity.json", process_identity())
    dump(owner / "registered_protocol.json", config)
    def unchanged():
        for group in ("source_bindings_sha256", "frozen_input_bindings_sha256"):
            for relative, expected in config[group].items():
                assert digest(root / relative) == expected, relative
    stable = None
    while True:
        assert shutil.disk_usage(root).free >= config["minimum_remaining_disk_bytes"]
        resources = memory()
        ready = resources["available_physical_bytes"] >= config["startup_available_physical_bytes"] and resources["remaining_commit_bytes"] >= config["startup_remaining_commit_bytes"]
        stable = (stable if stable is not None else time.monotonic()) if ready else None
        seconds = time.monotonic() - stable if stable is not None else 0.
        dump(owner / "progress.json", dict(utc=utc(), stage="WAIT_UNCHANGED_STARTUP_RESOURCE_GATES", resources=resources, stable_seconds=seconds))
        if seconds >= 15:
            break
        time.sleep(3)
    unchanged()
    with (owner / "native.log").open("x", encoding="utf-8") as log:
        child = launch_owned([str(NATIVE), str(script)], dict(os.environ), log, config["native_executable_sha256"])
        dump(owner / "launch_receipt.json", dict(native_identity=identity(child), registered_protocol_sha256=digest(owner / "registered_protocol.json")))
        print("ACTUAL337_OWNED_GEOMETRY_NATIVE_LAUNCHED", child.pid, script.name, flush=True)
        terminal = watch(child, output, config, owner / "resource_samples.jsonl", [])
    dump(owner / "actual_terminal.json", terminal)
    assert terminal["native_exit_code"] == terminal["launcher_exit_code"] == 0 and not terminal["resource_terminated"]
    result = read(output / "receipt.json")
    validate_result(result)
    assert result["NN_forwards"] == result["actual_optimizer_updates"] == 0
    unchanged()
    dump(owner / "receipt.json", dict(utc=utc(), actual_terminal=terminal,
        actual_result_sha256=digest(output / "receipt.json"), NN_forwards=0, actual_optimizer_updates=0, full_contract_pass=False))
    print("ACTUAL337_OWNED_GEOMETRY_NATURAL0", script.name, flush=True)
    return result
