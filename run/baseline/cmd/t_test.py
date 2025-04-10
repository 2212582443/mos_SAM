from typing import List
import torch
import scipy

from scipy.stats import shapiro, ttest_rel, pearsonr, t


def run(args: List[str] | None):
    print("args:", args)

    base_dir = ".checkpoint/baseline"

    data_a = torch.load(f"{base_dir}/segnet-2d-acdc,mnms/latest/test-metrics.pt")
    data_b = torch.load(f"{base_dir}/segnet-2d-none/latest/test-metrics.pt")

    for k in data_a.keys():
        print(f"\n-----------------------{k}----------------------------")
        a = data_a[k]
        b = data_b[k]

        df = a.shape[0] - 1

        print(
            "maen, std:\n\ta:",
            a.mean().item(),
            a.std().item(),
            "\n\tb:",
            b.mean().item(),
            b.std().item(),
        )
        print(
            "置信区间:\n\ta:",
            t.interval(0.95, df=df, loc=a.mean().item(), scale=a.std().item()),
            "\n\tb:",
            t.interval(0.95, df=df, loc=b.mean().item(), scale=b.std().item()),
        )
        print("均值差:\n\t", (a - b).mean().item())
        print("正态性：\n\ta:", shapiro(a), "\n\tb:", shapiro(b))
        print("t-test:\n\t", ttest_rel(a, b))
        print("pearsonr:\n\t", pearsonr(a, b))


# 参考：https://mengte.online/archives/5247
