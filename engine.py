"""
Antibiotic Policy Engine: deterministic, auditable core.

Merges three sources into one empiric recommendation per clinical syndrome:
  1. ICMR syndrome-wise regimens        (data/icmr_regimens.csv)
  2. The ICU's own cumulative antibiogram (data/<icu>_whonet_isolates.csv -> computed here)
  3. The hospital's stewardship policy    (data/policy.csv)

No machine learning makes the recommendation. Every number can be traced back
to a resistance percentage or a policy row, and the same input always gives
the same output.
"""
import json, hashlib
import pandas as pd
import numpy as np

SAFE_MAX, CAUTION_MAX = 10.0, 30.0          # % resistance thresholds for the three tiers
MIN_ISOLATES = 30                           # CLSI M39: fewer isolates -> estimate unreliable
TIER_NAMES = {1: 'SAFE', 2: 'USE WITH CAUTION', 3: 'NO RELIABLE EMPIRIC OPTION'}
AWARE_RANK = {'Access': 0, 'Watch': 1, 'Reserve': 2}
RESTRICT_RANK = {'free': 0, 'justify': 1, 'id_approval': 2}


# ---------------------------------------------------------------- antibiogram
def cumulative_antibiogram(isolates: pd.DataFrame) -> pd.DataFrame:
    """WHONET isolate export -> cumulative antibiogram (% resistant), CLSI M39 style.
    Keeps only the first isolate per patient per organism per specimen type."""
    first = isolates.sort_values('SPEC_DATE').drop_duplicates(['PATIENT_ID', 'ORGANISM', 'SPEC_TYPE'])
    abx_cols = [c for c in isolates.columns if c.isupper() and len(c) == 3 and c not in ('ICU',)]
    rows = []
    for (spec, org), g in first.groupby(['SPEC_TYPE', 'ORGANISM']):
        for a in abx_cols:
            v = g[a].dropna()
            if len(v) == 0:
                continue
            rows.append(dict(SPEC_TYPE=spec, ORGANISM=org, ANTIBIOTIC=a, N=len(v),
                             PCT_R=round(100 * (v != 'S').mean(), 1)))   # I counted as non-susceptible
    return pd.DataFrame(rows)


def organism_weights(isolates: pd.DataFrame) -> pd.DataFrame:
    """Share of each organism within a specimen type (used to weight resistance)."""
    first = isolates.drop_duplicates(['PATIENT_ID', 'ORGANISM', 'SPEC_TYPE'])
    w = first.groupby(['SPEC_TYPE', 'ORGANISM']).size().rename('N').reset_index()
    w['WEIGHT'] = w['N'] / w.groupby('SPEC_TYPE')['N'].transform('sum')
    return w


def regimen_resistance(isolates, spec, drugs):
    """Weighted-incidence syndromic resistance (WISCA idea), computed at isolate level:
    an isolate counts as covered if it is susceptible to AT LEAST ONE drug in the regimen.
    Untested drug/organism pairs (intrinsic resistance) count as not covered."""
    first = isolates.drop_duplicates(['PATIENT_ID', 'ORGANISM', 'SPEC_TYPE'])
    g = first[first.SPEC_TYPE == spec]
    if len(g) == 0:
        return None, 0, {}
    cov = np.zeros(len(g), bool)
    for d in drugs:
        if d in g.columns:
            cov |= (g[d] == 'S').values
    per_org = {}
    for org, gg in g.assign(_cov=cov).groupby('ORGANISM'):
        per_org[org] = round(100 * (1 - gg._cov.mean()), 1)
    return round(100 * (1 - cov.mean()), 1), len(g), per_org


def tier_of(pct_r):
    if pct_r is None: return 3
    return 1 if pct_r <= SAFE_MAX else (2 if pct_r <= CAUTION_MAX else 3)


# ---------------------------------------------------------------- engine
def recommend(syndrome, isolates, regimens, policy, specimen_map,
              exclude=(), patient_flags=()):
    """Return a ranked, fully explained recommendation for one syndrome in one ICU."""
    spec = specimen_map[syndrome]
    pol = policy.set_index('CODE')
    cands = regimens[regimens.SYNDROME == syndrome]
    out = []
    for _, r in cands.iterrows():
        drugs = r.DRUGS.split('+')
        reasons = []
        # patient-level exclusions (e.g. polymyxins in acute kidney injury, penicillin anaphylaxis)
        blocked = [d for d in drugs if d in exclude or any(f in pol.loc[d, 'CONTRAINDICATED_IN'].split('|') for f in patient_flags)]
        if blocked:
            out.append(dict(regimen=r.REGIMEN, drugs=drugs, line=r.LINE, status='EXCLUDED',
                            reason=f"contraindicated for this patient: {', '.join(pol.loc[b, 'NAME'] for b in blocked)}"))
            continue
        pct, n, per_org = regimen_resistance(isolates, spec, [d for d in drugs if pol.loc[d, 'CULTURE_EVALUABLE'] == 1])
        t = tier_of(pct)
        if n < MIN_ISOLATES:
            reasons.append(f'only {n} isolates (<{MIN_ISOLATES}): local estimate unreliable'); t = max(t, 2)
        aware = max((pol.loc[d, 'AWARE'] for d in drugs), key=lambda a: AWARE_RANK[a])
        restr = max((pol.loc[d, 'RESTRICTION'] for d in drugs), key=lambda a: RESTRICT_RANK[a])
        avail = all(pol.loc[d, 'FORMULARY'] == 1 for d in drugs)
        if not avail:
            out.append(dict(regimen=r.REGIMEN, drugs=drugs, line=r.LINE, status='EXCLUDED',
                            reason='not on hospital formulary')); continue
        if restr == 'id_approval': reasons.append('requires Infectious Disease approval')
        elif restr == 'justify': reasons.append('prescriber justification required')
        out.append(dict(regimen=r.REGIMEN, drugs=drugs, line=r.LINE, status='CANDIDATE',
                        pct_resistant=pct, isolates=n, per_organism=per_org, tier=t, tier_name=TIER_NAMES[t],
                        aware=aware, restriction=restr, reason='; '.join(reasons), dosing=r.DOSING))
    ok = [c for c in out if c['status'] == 'CANDIDATE']
    # stewardship ranking: safest tier, then narrowest spectrum (AWaRe), least restricted, fewest drugs, best coverage
    ok.sort(key=lambda c: (c['tier'], AWARE_RANK[c['aware']], RESTRICT_RANK[c['restriction']], len(c['drugs']), c['pct_resistant']))
    best = ok[0] if ok else None
    if best is None or best['tier'] == 3:
        decision = dict(action='ESCALATE',
                        message='No regimen in the guideline is reliably active in this ICU. Do not start a '
                                'potentially inactive empiric regimen: send cultures and call Microbiology / '
                                'Infectious Disease now.')
    else:
        decision = dict(action='RECOMMEND', regimen=best['regimen'], tier=best['tier_name'],
                        message=f"{best['regimen']}: {best['pct_resistant']}% of local {spec.lower()} isolates resistant"
                                + (f" ({best['reason']})" if best['reason'] else ''))
    result = dict(syndrome=syndrome, specimen=spec, decision=decision, ranked=ok,
                  excluded=[c for c in out if c['status'] == 'EXCLUDED'],
                  patient_flags=list(patient_flags))
    result['audit_hash'] = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return result


def pretty(res):
    lines = [f"\n=== {res['syndrome']}  (specimen: {res['specimen']}) ===",
             f"DECISION: {res['decision']['action']} | {res['decision']['message']}"]
    for i, c in enumerate(res['ranked'], 1):
        lines.append(f"  {i}. {c['regimen']:<45} {c['pct_resistant']:>5}% R  [{c['tier_name']}]  {c['aware']}"
                     + (f"  - {c['reason']}" if c['reason'] else ''))
    for c in res['excluded']:
        lines.append(f"  x  {c['regimen']:<45} EXCLUDED - {c['reason']}")
    lines.append(f"  audit hash: {res['audit_hash']}")
    return '\n'.join(lines)
