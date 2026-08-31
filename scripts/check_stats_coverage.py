import json, glob, re
from collections import defaultdict

EVO = ['topological_shift', 'strategic_shift', '3-gram_wasserstein']


def has_null(d):
    v = d.get('zmax_p95')
    if not (isinstance(v, list) and v and any(x is not None for x in v)):
        return False
    for k in EVO:
        m = d.get('null_mean_' + k)
        s = d.get('null_std_' + k)
        if not (isinstance(m, list) and m and any(x is not None for x in m)):
            return False
        if not (isinstance(s, list) and s and any(x is not None for x in s)):
            return False
    return True


# file-level classification, grouped by cell (env/model/mode)
cells = defaultdict(lambda: {'with': 0, 'without': 0})
for f in glob.glob('results/**/*_metrics.json', recursive=True):
    n = f.replace('\\', '/').split('/')
    key = (n[1], n[2], n[3])
    try:
        d = json.load(open(f))
    except Exception:
        continue
    cells[key]['with' if has_null(d) else 'without'] += 1

ok_cells, redo_cells = [], []
for k in sorted(cells):
    c = cells[k]
    (ok_cells if c['with'] > 0 else redo_cells).append((k, c))

print("=== CELLS WITH USABLE FAMILY-WISE STATS (>=1 file) ===")
for k, c in ok_cells:
    print(f"  {k[0]}/{k[1]}/{k[2]}: {c['with']} with stats, {c['without']} without")

print("\n=== CELLS NEEDING REGENERATION (NO file has family-wise stats) ===")
for k, c in redo_cells:
    print(f"  {k[0]}/{k[1]}/{k[2]}: {c['without']} file(s), none with stats")

print(f"\n{len(ok_cells)} cell(s) usable, {len(redo_cells)} cell(s) need redo.")
