from typing import Callable, Generic, Iterable, Iterator, List, Tuple, TypeGuard, TypeVar, overload


_T = TypeVar("_T")
_S = TypeVar("_S")


def bi_partition(function: Callable[[_S], TypeGuard[_T]], iterable: Iterable[_S]) -> Tuple[List[_S], List[_S]]:
    """对一个list进行二分类, 返回两个list, 第一个list是true, 第二个list是false

    bi partion a list by function,
    return two list,
        first list is true,
        second list is false
    """
    true_list = []
    false_list = []

    for item in iterable:
        if function(item):
            true_list.append(item)
        else:
            false_list.append(item)

    return (true_list, false_list)
