import json
import subprocess
import threading


def test_mcp():
    print("Launching MCP Server...")
    p = subprocess.Popen(
        [
            "/sep/structural-manifold-compression/.venv/bin/python",
            "/sep/structural-manifold-compression/mcp_server.py",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    }

    payload = json.dumps(init_req)
    msg = f"Content-Length: {len(payload)}\r\n\r\n{payload}"

    p.stdin.write(msg)
    p.stdin.flush()

    def read_stderr():
        while True:
            err = p.stderr.readline()
            if not err:
                break
            print(f"STDERR: {err}", end="")

    t = threading.Thread(target=read_stderr, daemon=True)
    t.start()

    try:
        out1 = p.stdout.readline()
        print(f"STDOUT1: {out1!r}")
        out2 = p.stdout.readline()
        print(f"STDOUT2: {out2!r}")
        out3 = p.stdout.readline()
        print(f"STDOUT3: {out3!r}")
        out4 = p.stdout.readline()
        print(f"STDOUT4: {out4!r}")
    except Exception as e:
        pass

    p.terminate()


if __name__ == "__main__":
    test_mcp()
