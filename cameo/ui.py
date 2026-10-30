"""Turn an API-format graph into a frontend workflow (editable in ComfyUI, listed under Workflows),
using comfy-cli's own edit ops so node layouts and widgets match the live catalog.

    python -m cameo.ui workflows/cameo_insert_api.json workflows/cameo_insert.json
"""
import json
import subprocess
import sys

EMPTY = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [],
         "config": {}, "extra": {}, "version": 0.4}
CATALOG = ["--where", "local", "--host", "127.0.0.1", "--port", "8188"]


def op(*args):
    out = subprocess.run(["comfy", "workflow", *args, *CATALOG], capture_output=True, text=True)
    env = json.loads(out.stdout.splitlines()[0])
    if not env["ok"]:
        raise RuntimeError(f"{args}: {env['error']['message']}")
    return env["data"]


def convert(api, path):
    json.dump(EMPTY, open(path, "w"))
    ids, pos = {}, {}
    # lay nodes out left to right by dependency depth
    def depth(nid, seen=()):
        links = [v for v in api[nid]["inputs"].values() if isinstance(v, list)]
        return 0 if not links else 1 + max(depth(v[0], seen + (nid,)) for v in links)
    rows = {}
    for nid in api:
        d = depth(nid)
        pos[nid] = f"{40 + d * 360},{60 + rows.get(d, 0) * 220}"
        rows[d] = rows.get(d, 0) + 1
    for nid, node in api.items():
        data = op("add-node", path, node["class_type"], "--at", pos[nid])
        ids[nid] = data["op"]["node_id"]
    for nid, node in api.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                op("connect", path, f"{ids[value[0]]}.{value[1]}", f"{ids[nid]}.{name}")
            else:
                op("set-widget", path, f"{ids[nid]}.{name}", json.dumps(value))
    return path


if __name__ == "__main__":
    convert(json.load(open(sys.argv[1])), sys.argv[2])
    print("wrote", sys.argv[2])
