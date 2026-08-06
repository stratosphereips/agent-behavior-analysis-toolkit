import json, glob, re
from statistics import mean, pstdev

EXCLUDE_SUBSTR = ['evaleach_1500', 'seed_comparison']

def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return None

def arr(d, k):
    v = d.get(k)
    return v if isinstance(v, list) else []

def frac_active(d, raw, thr):
    r, t = arr(d, raw), arr(d, thr)
    n = min(len(r), len(t))
    if n == 0:
        return None
    return sum(1 for i in range(n) if r[i] > t[i]) / n

def conv_checkpoint(d):
    cks = arr(d, 'checkpoints')
    rr = {k: arr(d, k + '_raw') for k in ['topological_shift', 'strategic_shift', '3-gram_wasserstein']}
    tt = {k: arr(d, k + '_noise_threshold') for k in ['topological_shift', 'strategic_shift', '3-gram_wasserstein']}
    last = -1
    for i in range(len(cks)):
        if any(i < len(rr[k]) and i < len(tt[k]) and rr[k][i] > tt[k][i] for k in rr):
            last = i
    if last < 0 or last >= len(cks) - 2:
        return None  # never
    return cks[last + 1]

files = glob.glob('results/frozenlake8x8/**/*_metrics.json', recursive=True)
files = [f for f in files if not any(x in f for x in EXCLUDE_SUBSTR)]

# disambiguate seed from FILENAME first (handles two seeds sharing one folder), else folder
groups = {}
for f in files:
    norm = f.replace('\\', '/')
    parts = norm.split('/')
    algo = parts[2] if len(parts) > 2 else '?'
    mode = parts[3] if len(parts) > 3 else '?'
    fname = parts[-1]
    m = re.search(r'seed[=_]?(\d+)', fname) or re.search(r'seed[=_]?(\d+)', norm)
    seed = m.group(1) if m else ''
    mm = re.search(r'_(\d+)_metrics\.json$', norm)
    M = int(mm.group(1)) if mm else -1
    key = (algo, mode, seed, norm)  # keep filename in key to avoid collapsing distinct seeds
    groups[key] = (M, f)

# now re-key by (algo,mode,seed) picking max-M file, but seed now correctly disambiguated
final = {}
for (algo, mode, seed, norm), (M, f) in groups.items():
    key = (algo, mode, seed)
    prev = final.get(key)
    if prev is None or M > prev[0]:
        final[key] = (M, f)

rows = []
for (algo, mode, seed), (M, f) in sorted(final.items()):
    d = load(f)
    if not d:
        continue
    Seff = arr(d, 'state_visitation_perplexity')
    V = arr(d, 'total_nodes')
    ret = arr(d, 'mean_return')
    rtrue = arr(d, 'mean_r_true')
    pTopo = frac_active(d, 'topological_shift_raw', 'topological_shift_noise_threshold')
    pStrat = frac_active(d, 'strategic_shift_raw', 'strategic_shift_noise_threshold')
    conv = conv_checkpoint(d)
    rows.append(dict(algo=algo, mode=mode, seed=seed,
                      V=V[-1] if V else None, Seff=Seff[-1] if Seff else None,
                      pTopo=pTopo, pStrat=pStrat, ret=ret[-1] if ret else None,
                      rtrue=rtrue[-1] if rtrue else None, conv=conv))

print(f"{'algo':10}{'mode':20}{'seed':6}{'|V|':>5}{'Seff':>7}{'%To':>6}{'%St':>6}{'ret':>8}{'r_true':>8}{'conv':>7}")
for r in sorted(rows, key=lambda r: (r['algo'], r['mode'], r['seed'])):
    print(f"{r['algo']:10}{r['mode']:20}{r['seed']:6}{str(round(r['V']) if r['V'] else None):>5}"
          f"{str(round(r['Seff'],1) if r['Seff'] else None):>7}{str(round(r['pTopo'],2) if r['pTopo'] is not None else None):>6}"
          f"{str(round(r['pStrat'],2) if r['pStrat'] is not None else None):>6}{str(round(r['ret'],3) if r['ret'] is not None else None):>8}"
          f"{str(round(r['rtrue'],4) if r['rtrue'] is not None else None):>8}{str(r['conv'] or 'never'):>7}")

print(f"\ntotal rows: {len(rows)}\n")
print("=== AGGREGATE (mean +/- std, n) for TABLE ===")
agg = {}
for r in rows:
    agg.setdefault((r['algo'], r['mode']), []).append(r)
for (algo, mode), items in sorted(agg.items()):
    n = len(items)
    def col(k, nd=1):
        xs = [i[k] for i in items if i[k] is not None]
        if not xs: return 'NA'
        return f"{mean(xs):.{nd}f}" + (f"+/-{pstdev(xs):.{nd}f}" if n > 1 else "")
    nevers = sum(1 for i in items if i['conv'] is None)
    conv_note = 'never' if nevers >= n / 2 else str(round(mean([i['conv'] for i in items if i['conv'] is not None])))
    print(f"{algo:10}{mode:20} n={n:2}  |V|={col('V',0):12} Seff={col('Seff',1):12} "
          f"%To={col('pTopo',2):12} %St={col('pStrat',2):12} ret={col('ret',3):14} r_true={col('rtrue',4):10} conv~{conv_note}")
