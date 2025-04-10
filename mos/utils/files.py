import os, errno


class FileInfo(object):
    file_abs_path: str
    root: str
    name: str

    def __init__(self, file_abs_path: str, root: str, name: str):
        self.file_abs_path = file_abs_path
        self.root = root
        self.name = name


def load_files(dir: str, ext: str) -> list[FileInfo]:
    """
    加载目录下的所有文件
    """
    files_list = []  # (file, root, name)
    for root, _dirs, files in os.walk(dir, topdown=False):
        for name in files:
            f = os.path.join(root, name)
            if name.endswith(ext):
                file_info = FileInfo(f, root, name)
                files_list.append(file_info)

    files_list.sort(key=lambda x: x.file_abs_path)
    return files_list


def get_file_ext(file: str) -> str:
    """
    获取文件扩展名(包含.)
    """
    ext = ""
    if file.endswith(".gz"):
        ext = ".gz"
        file = file.rstrip(".gz")

    ext = os.path.splitext(file)[1].lower() + ext

    return ext


def relative_symlink_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    try:
        os.remove(dst)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred
    dir = os.path.dirname(dst)
    src = os.path.relpath(src, dir)
    dst = os.path.join(dir, os.path.basename(dst))
    return os.symlink(src, dst)
