#!/usr/bin/env python3
"""Watch redis INSTANCES_MAP writes during a pytest run (redis-py MONITOR).

Self-discovers the testcontainers redis port via docker ps (polls until a
redis container appears), then tails MONITOR for solar:hosts:* writes.

Usage: watch_redis.py <outfile>
"""

import subprocess
import sys
import time

out = sys.argv[1]

port = None
for _ in range(120):
    try:
        rows = subprocess.run(
            ["docker", "ps", "--format", "{{.Image}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except Exception:
        rows = []
    for row in rows:
        if "redis" in row.lower():
            for part in row.split("\t")[1].split(","):
                if "6379" in part and "->" in part:
                    port = part.split("->")[0].split(":")[-1].strip()
                    break
        if port:
            break
    if port:
        break
    time.sleep(1)

if not port:
    print("no redis container found", file=sys.stderr)
    sys.exit(1)

import redis  # noqa: E402

r = redis.Redis(host="127.0.0.1", port=int(port), socket_timeout=5)
with open(out, "a") as f:
    f.write(f"=== monitor start port={port} {time.time():.3f} ===\n")
    f.flush()
    try:
        with r.monitor() as m:
            for line in m.listen():
                if "solar:hosts" in str(line):
                    f.write(f"{time.time():.3f} {line}\n")
                    f.flush()
    except KeyboardInterrupt:
        pass
