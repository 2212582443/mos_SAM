import os


_SEARCH_PATHS = [
    os.path.expanduser("~"),
    os.path.join(os.path.expanduser("~"), "dataset"),
    os.path.join("/data", os.environ["USER"]),
    os.path.join("/data", os.environ["USER"], "dataset"),
]


def locate_dataset_base_url(base_url: str) -> str:
    """定位数据集根目录

    1. 如果base_url是绝对路径, 则直接返回base_url
    2. 如果base_url是相对路径而且存在, 则返回当前工作目录下的base_url的绝对路径()
    3. 如果~/base_url存在, 则返回~/base_url的绝对路径

    """
    if base_url.startswith("~/"):
        base_url = os.path.expanduser(base_url)

    if os.path.isabs(base_url):
        return base_url

    if os.path.exists(base_url):
        return os.path.abspath(base_url)

    for search_path in _SEARCH_PATHS:
        search_base_url = os.path.join(search_path, base_url)
        if os.path.exists(search_base_url):
            return os.path.abspath(search_base_url)

    raise ValueError(f"base_url {base_url} not found")
