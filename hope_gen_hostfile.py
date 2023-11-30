# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
    Copyright @2023 GUANGNIANAI Inc. (guannianai.com)

    @author: Q.Y.Duan <duanqiyuan@guangnianai.com>
    @date: 2023/11/30
"""
import argparse
import os
import json


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", type=str, help="hostfile 文件保存路径")
    return parser


def gen_host_file(target_path):
    cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    worker_list = cluster_spec["worker"]
    with open(target_path, "w") as f:
        for host in worker_list:
            message = f"{host.split(':')[0]} port=8022 slots=8\n"
            f.write(message)


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()
    gen_host_file(args.target_path)
