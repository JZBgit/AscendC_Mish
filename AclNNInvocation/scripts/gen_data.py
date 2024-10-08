#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

def gen_golden_data_simple():
    input_x = np.random.uniform(1, 10, [8, 2048]).astype(np.float16)
    # 生成Mish测试数据
    golden = input_x*np.tanh(np.log(1+np.exp(input_x)))

    # print(golden)
    input_x.tofile("./AclNNInvocation/input/input_x.bin")
    golden.tofile("./AclNNInvocation/output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
