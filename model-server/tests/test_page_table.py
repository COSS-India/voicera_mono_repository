"""Coverage for the Parler KV page allocator.

PageTable is the one part of the inference engine that is pure Python, so it can
be tested without a GPU. It is also the part where a subtle bug is worst: two
requests handed the same page means silent KV-cache corruption that only shows
up under concurrency.

The class is extracted from the real paging.py by AST, so the test fails if the
source drifts rather than quietly passing against a stale copy.
"""
import ast
import math
import random
from pathlib import Path

import pytest

PAGING = Path(__file__).resolve().parent.parent / "tts" / "inference" / "paging.py"


def _load_page_table():
    tree = ast.parse(PAGING.read_text(encoding="utf-8").replace("\r\n", "\n"))
    node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PageTable"),
        None,
    )
    assert node is not None, "paging.py no longer defines PageTable"
    ns = {"math": math}
    exec(compile(ast.Module([node], []), "<paging>", "exec"), ns)  # noqa: S102
    return ns["PageTable"]


PageTable = _load_page_table()


def _pages_are_exclusive(table):
    seen = set()
    for pages in table.pid_page_table.values():
        for page in pages:
            assert page not in seen, f"page {page} handed to two requests"
            seen.add(page)
    return seen


def test_allocation_rounds_up_to_page_size():
    t = PageTable(page_size=8)
    t.allocate("a", 1)
    assert len(t.pid_page_table["a"]) == 1
    t.allocate("a", 8)
    assert len(t.pid_page_table["a"]) == math.ceil(9 / 8)


def test_pages_are_never_shared_between_requests():
    t = PageTable(page_size=4)
    for pid in ("a", "b", "c"):
        t.allocate(pid, 30)
    _pages_are_exclusive(t)


def test_freed_gaps_are_reused_before_extending():
    t = PageTable(page_size=1)
    for pid in ("a", "b", "c"):
        t.allocate(pid, 4)
    highest_before = max(p for pages in t.pid_page_table.values() for p in pages)
    # drop the middle request, leaving a hole
    del t.pid_page_table["b"]
    del t.pid_mem_sizes["b"]
    t.allocate("d", 4)
    highest_after = max(p for pages in t.pid_page_table.values() for p in pages)
    assert highest_after == highest_before, "allocator grew instead of reusing the gap"
    _pages_are_exclusive(t)


def test_budget_is_enforced():
    t = PageTable(page_size=1, max_num_pages=4)
    t.allocate("a", 4)
    with pytest.raises(RuntimeError, match=r'paged KV overflow'):
        t.allocate("b", 4)


@pytest.mark.parametrize("seed", range(25))
def test_randomised_sequences_keep_pages_exclusive(seed):
    rng = random.Random(seed)
    t = PageTable(page_size=rng.choice([1, 4, 8, 16]), max_num_pages=4096)
    for _ in range(rng.randint(1, 30)):
        t.allocate(f"p{rng.randint(0, 7)}", rng.randint(1, 40))
        _pages_are_exclusive(t)
