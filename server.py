#!/usr/bin/env python3
"""Serve an OpenJevX checkpoint through Laya's Jev-compatible HTTP API."""

import argparse

import laya
import uvicorn
from laya import Router
from laya.serve import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="muthuishere/openjevx")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()

    agent = laya.load(args.model, device=args.device, compile=False)
    router = Router(max_loaded=3, default="english", preload=False)
    for name in ("english", "multilingual", "typed-decisions"):
        router.attach(name, agent)
    print(f"OpenJevX ready at http://{args.host}:{args.port}/v1/systemone", flush=True)
    uvicorn.run(create_app(router), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
