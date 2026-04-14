"""ARC grid-primitive library (the AlphaProgram DSL).

The DSL is restricted Python: proposers write a `def transform(grid)`
function and may use only:
  - the standard Python control flow (if, for, while, list/dict comprehensions)
  - the helpers exported from this module
  - basic arithmetic and comparisons
  - list/tuple/set/dict, len, range, min, max, sum, sorted, abs, any, all

The verifier (verifier.py) imports this module into a sandboxed namespace
and executes the proposer's `transform` function on the task's training
input/output pairs, scoring how many pairs match exactly.

This module deliberately avoids being too clever. It exposes ~30 small
primitives that cover the bulk of ARC public-train transformations:
  - Geometric: rotate, flip_h, flip_v, transpose, crop, pad, tile
  - Color: recolor, swap_colors, palette, count_colors
  - Object: find_objects, bounding_box, fill, flood_fill
  - Construction: grid_of, copy_grid, paint, shape, h, w
  - Arithmetic on grids: zeros_like, ones_like
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

# A grid is a list[list[int]]. We use plain lists rather than numpy so
# proposers don't need to know array semantics.
Grid = list[list[int]]


# ───── geometry ─────────────────────────────────────────────────────────────

def h(g: Grid) -> int:
    """Height (number of rows) of a grid."""
    return len(g)


def w(g: Grid) -> int:
    """Width (number of columns) of a grid."""
    return len(g[0]) if g else 0


def shape(g: Grid) -> tuple[int, int]:
    """(height, width) of a grid."""
    return (h(g), w(g))


def copy_grid(g: Grid) -> Grid:
    """Deep copy a grid."""
    return [row[:] for row in g]


def grid_of(rows: int, cols: int, value: int = 0) -> Grid:
    """Create a `rows × cols` grid filled with `value`."""
    return [[value] * cols for _ in range(rows)]


def zeros_like(g: Grid) -> Grid:
    """A grid the same shape as `g`, filled with 0."""
    return grid_of(h(g), w(g), 0)


def ones_like(g: Grid) -> Grid:
    """A grid the same shape as `g`, filled with 1."""
    return grid_of(h(g), w(g), 1)


def rotate(g: Grid, k: int = 1) -> Grid:
    """Rotate a grid 90° clockwise k times. k may be negative."""
    g = copy_grid(g)
    k = k % 4
    for _ in range(k):
        g = [[g[h(g) - 1 - r][c] for r in range(h(g))] for c in range(w(g))]
    return g


def flip_h(g: Grid) -> Grid:
    """Flip horizontally (mirror left-right)."""
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    """Flip vertically (mirror top-bottom)."""
    return g[::-1]


def transpose(g: Grid) -> Grid:
    """Transpose: rows become columns."""
    return [[g[r][c] for r in range(h(g))] for c in range(w(g))]


def crop(g: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Crop the rectangle [r0, r1) x [c0, c1)."""
    return [row[c0:c1] for row in g[r0:r1]]


def pad(g: Grid, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, value: int = 0) -> Grid:
    """Pad with `value` on each side."""
    new_w = w(g) + left + right
    out = []
    for _ in range(top):
        out.append([value] * new_w)
    for row in g:
        out.append([value] * left + row[:] + [value] * right)
    for _ in range(bottom):
        out.append([value] * new_w)
    return out


def tile(g: Grid, rows: int, cols: int) -> Grid:
    """Tile a grid `rows` x `cols` times."""
    out = []
    for _ in range(rows):
        for row in g:
            out.append(row * cols)
    return out


# ───── color ────────────────────────────────────────────────────────────────

def recolor(g: Grid, mapping: dict[int, int]) -> Grid:
    """Apply a color mapping. Colors not in `mapping` are kept as-is."""
    return [[mapping.get(c, c) for c in row] for row in g]


def swap_colors(g: Grid, a: int, b: int) -> Grid:
    """Swap two colors."""
    return recolor(g, {a: b, b: a})


def palette(g: Grid) -> set[int]:
    """The set of distinct colors in a grid."""
    return {c for row in g for c in row}


def count_color(g: Grid, color: int) -> int:
    """Count cells of a given color."""
    return sum(1 for row in g for c in row if c == color)


def most_common_color(g: Grid) -> int:
    """Color that appears the most. Ties broken by lowest color value."""
    counts: dict[int, int] = {}
    for row in g:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return min(counts, key=lambda k: (-counts[k], k))


def least_common_color(g: Grid) -> int:
    """Color that appears the least (and at least once). Ties → lowest value."""
    counts: dict[int, int] = {}
    for row in g:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return min(counts, key=lambda k: (counts[k], k))


# ───── object detection ────────────────────────────────────────────────────

def find_objects(g: Grid, background: int = 0, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    """Connected components of non-background cells.

    Returns a list of objects; each object is a list of (row, col) cells.
    `connectivity` is 4 (orthogonal) or 8 (orthogonal + diagonal).
    """
    H, W = h(g), w(g)
    seen = [[False] * W for _ in range(H)]
    nbrs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nbrs8 = nbrs4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    nbrs = nbrs8 if connectivity == 8 else nbrs4

    objs = []
    for r in range(H):
        for c in range(W):
            if seen[r][c] or g[r][c] == background:
                continue
            color = g[r][c]
            stack = [(r, c)]
            obj: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if seen[rr][cc] or g[rr][cc] != color:
                    continue
                seen[rr][cc] = True
                obj.append((rr, cc))
                for dr, dc in nbrs:
                    stack.append((rr + dr, cc + dc))
            objs.append(obj)
    return objs


def bounding_box(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    """(r0, c0, r1, c1) bounding box of a set of cells. r1, c1 are exclusive."""
    cells = list(cells)
    if not cells:
        return (0, 0, 0, 0)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (min(rs), min(cs), max(rs) + 1, max(cs) + 1)


def fill(g: Grid, color: int) -> Grid:
    """Return a grid of the same shape filled with `color`."""
    return grid_of(h(g), w(g), color)


def flood_fill(g: Grid, r: int, c: int, new_color: int) -> Grid:
    """Flood-fill the connected region starting at (r,c) with `new_color`."""
    g = copy_grid(g)
    H, W = h(g), w(g)
    if r < 0 or r >= H or c < 0 or c >= W:
        return g
    target = g[r][c]
    if target == new_color:
        return g
    q = deque([(r, c)])
    while q:
        rr, cc = q.popleft()
        if rr < 0 or rr >= H or cc < 0 or cc >= W:
            continue
        if g[rr][cc] != target:
            continue
        g[rr][cc] = new_color
        q.extend([(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)])
    return g


def paint(g: Grid, cells: Iterable[tuple[int, int]], color: int) -> Grid:
    """Paint the given cells in a copy of `g` with `color`."""
    g = copy_grid(g)
    H, W = h(g), w(g)
    for r, c in cells:
        if 0 <= r < H and 0 <= c < W:
            g[r][c] = color
    return g


# ───── inspection ──────────────────────────────────────────────────────────

def get(g: Grid, r: int, c: int, default: int = 0) -> int:
    """Safe element access."""
    if 0 <= r < h(g) and 0 <= c < w(g):
        return g[r][c]
    return default


def row(g: Grid, r: int) -> list[int]:
    """The r-th row (a copy)."""
    return g[r][:]


def col(g: Grid, c: int) -> list[int]:
    """The c-th column."""
    return [g[r][c] for r in range(h(g))]


# ───── the safe namespace exposed to proposers ─────────────────────────────

DSL_NAMESPACE = {
    # geometry
    "h": h, "w": w, "shape": shape, "copy_grid": copy_grid,
    "grid_of": grid_of, "zeros_like": zeros_like, "ones_like": ones_like,
    "rotate": rotate, "flip_h": flip_h, "flip_v": flip_v, "transpose": transpose,
    "crop": crop, "pad": pad, "tile": tile,
    # color
    "recolor": recolor, "swap_colors": swap_colors, "palette": palette,
    "count_color": count_color, "most_common_color": most_common_color,
    "least_common_color": least_common_color,
    # objects
    "find_objects": find_objects, "bounding_box": bounding_box,
    "fill": fill, "flood_fill": flood_fill, "paint": paint,
    # inspection
    "get": get, "row": row, "col": col,
    # plain Python builtins the proposer is allowed to use
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "abs": abs, "any": any, "all": all,
    "list": list, "tuple": tuple, "set": set, "dict": dict,
    "int": int, "str": str, "bool": bool,
    "enumerate": enumerate, "zip": zip, "reversed": reversed,
    "True": True, "False": False, "None": None,
    "print": lambda *a, **k: None,  # silenced print
}


def dsl_doc() -> str:
    """A short docstring used in the proposer prompt to teach the DSL."""
    return DSL_DOC


DSL_DOC = """\
You may use the following helpers in your `transform(grid)` function. A grid is
a `list[list[int]]` (rows of integers, each integer is a color 0-9).

GEOMETRY:
  h(g)               -> int             height (rows)
  w(g)               -> int             width (cols)
  shape(g)           -> (h, w)          (height, width)
  copy_grid(g)       -> Grid            deep copy
  grid_of(r, c, v=0) -> Grid            new grid filled with v
  zeros_like(g)      -> Grid            same shape, all zeros
  ones_like(g)       -> Grid            same shape, all ones
  rotate(g, k=1)     -> Grid            rotate 90 deg clockwise k times
  flip_h(g)          -> Grid            mirror left-right
  flip_v(g)          -> Grid            mirror top-bottom
  transpose(g)       -> Grid            rows <-> cols
  crop(g, r0, c0, r1, c1) -> Grid       crop [r0:r1, c0:c1)
  pad(g, top, bot, left, right, value=0) -> Grid
  tile(g, rows, cols) -> Grid           tile g into a rows x cols meta-grid

COLOR:
  recolor(g, mapping)        -> Grid    apply {old: new, ...}; unmapped kept
  swap_colors(g, a, b)       -> Grid    swap two colors
  palette(g)                 -> set     distinct colors
  count_color(g, color)      -> int     number of cells of that color
  most_common_color(g)       -> int
  least_common_color(g)      -> int

OBJECTS:
  find_objects(g, background=0, connectivity=4) -> list[list[(r,c)]]
      connected components of non-background cells; each object is its cells
  bounding_box(cells)        -> (r0, c0, r1, c1)   r1, c1 exclusive
  fill(g, color)             -> Grid    same shape, all `color`
  flood_fill(g, r, c, new)   -> Grid    flood-fill from (r,c)
  paint(g, cells, color)     -> Grid    paint a list of (r,c) cells

INSPECTION:
  get(g, r, c, default=0)    -> int     safe element access
  row(g, r) / col(g, c)      -> list[int]

You also have plain Python: if/for/while, list/dict comprehensions, len, range,
min, max, sum, sorted, abs, any, all, list/tuple/set/dict, enumerate, zip.

You MUST define `def transform(grid):` returning a new grid (list[list[int]]).
You may NOT use imports, file I/O, eval, exec, or any I/O. Stay pure.
"""
