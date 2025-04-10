#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-05-03
# @Author  : yehuping

import importlib, sys, os

importlib.invalidate_caches()


if __name__ == "__main__":
    argv: list[str] = sys.argv
    if len(argv) < 2:
        print("Please input command")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    package = argv.pop(1)
    if package == "" or package[:1] == "-":
        print("Please input command: run.py <command>")
        exit(1)

    if package.startswith("."):
        package = f"run{package}"

    run_module = importlib.import_module(package, ".")
    argv.pop(0)
    run_module.run(argv)
