from typing import List
from mos.core.opts import DefaultAppArgs


def run(args: List[str] | None):
    """ Test run script

    ```bash
        bins/run run.test --seed 9998
    ```
    """
    args = DefaultAppArgs.from_args(args)
    print("Test, args:", args)
