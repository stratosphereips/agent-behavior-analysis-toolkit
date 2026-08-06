import json, glob, re

rows = []
for f in glob.glob('results/**/*_metrics.json', recursive=True):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    parts = f.replace('\\', '/').split('/')
    env = parts[1]
    algo = parts[2] if len(parts) > 2 else '?'
    mode = parts[3] if len(parts) > 3 else '?'
    m = re.search(r'seed[_]?(\w+)', f)
    seed = m.group(1) if m else ''

    def arr(k):
        v = d.get(k)
        return v if isinstance(v, list) else []

    Seff = arr('state_visitation_perplexity')
    V = arr('total_nodes')
    ret = arr('mean_return')

    def frac_active(raw, thr):
        r, t = arr(raw), arr(thr)
        n = min(len(r), len(t))
        if n == 0:
            return None
        return sum(1 for i in range(n) if r[i] > t[i]) / n

    pTopo = frac_active('topological_shift_raw', 'topological_shift_noise_threshold')
    pStrat = frac_active('strategic_shift_raw', 'strategic_shift_noise_threshold')

    cks = arr('checkpoints')
    rr = {k: arr(k + '_raw') for k in ['topological_shift', 'strategic_shift', '3-gram_wasserstein']}
    tt = {k: arr(k + '_noise_threshold') for k in ['topological_shift', 'strategic_shift', '3-gram_wasserstein']}
    last = -1
    for i in range(len(cks)):
        act = any(i < len(rr[k]) and i < len(tt[k]) and rr[k][i] > tt[k][i] for k in rr)
        if act:
            last = i
    conv = 'never' if last >= len(cks) - 2 or last < 0 else str(cks[last + 1])

    rows.append((env, algo, mode, seed,
                 round(V[-1]) if V else None,
                 round(Seff[-1]) if Seff else None,
                 round(pTopo, 2) if pTopo is not None else None,
                 round(pStrat, 2) if pStrat is not None else None,
                 round(ret[-1], 1) if ret else None,
                 conv))

rows.sort()
hdr = f"{'env':14}{'algo':10}{'mode':20}{'seed':7}{'|V|':>5}{'Seff':>6}{'%To':>6}{'%St':>6}{'ret':>9}{'conv':>8}"
print(hdr)
for r in rows:
    print(f"{r[0]:14}{r[1]:10}{r[2]:20}{str(r[3]):7}{str(r[4]):>5}{str(r[5]):>6}{str(r[6]):>6}{str(r[7]):>6}{str(r[8]):>9}{str(r[9]):>8}")
print(f"\nTOTAL metrics.json files: {len(rows)}")
