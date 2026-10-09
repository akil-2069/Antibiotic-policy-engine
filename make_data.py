"""
Builds the input files for the Antibiotic Policy Engine.

- data/icmr_regimens.csv      syndrome-wise empiric regimens curated from the ICMR
                              Treatment Guidelines for Antimicrobial Use in Common Syndromes (2019)
- data/specimen_map.csv       syndrome -> clinical specimen type
- data/policy.csv             hospital stewardship policy (AWaRe class, restriction, formulary, contraindications)
- data/icu_A_whonet.csv,      synthetic WHONET-style isolate exports for two ICUs
  data/icu_B_whonet.csv       (A = tertiary, high resistance; B = secondary, lower resistance)

The resistance rates are SYNTHETIC. They are set to roughly the ranges published in
ICMR AMRSN annual reports for Indian ICUs, so the engine behaves realistically, but
they are not any real hospital's data. Replace the two WHONET files with a real export
to run the engine for a real ICU.
"""
import os
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ ICMR regimens
REG = [
 # syndrome, line, regimen, drugs, dosing (adult, normal renal function), source
 ('Acute undifferentiated fever (severe)', 1, 'Ceftriaxone + Doxycycline', 'CRO+DOX', 'Ceftriaxone 2 g IV q24h + Doxycycline 100 mg IV/PO q12h', 'ICMR 2019 Ch.2 (enteric fever + scrub typhus cover)'),
 ('Acute undifferentiated fever (severe)', 2, 'Piperacillin-tazobactam + Doxycycline', 'TZP+DOX', 'Pip-tazo 4.5 g IV q6h + Doxycycline 100 mg q12h', 'ICMR 2019 Table 2.1'),
 ('Acute undifferentiated fever (severe)', 3, 'Meropenem + Doxycycline', 'MEM+DOX', 'Meropenem 1 g IV q8h + Doxycycline 100 mg q12h', 'ICMR 2019 Table 2.1 (seriously ill)'),
 ('Sepsis / septic shock (unclear focus)', 1, 'Imipenem + Vancomycin', 'IPM+VAN', 'Imipenem 1 g IV q8h + Vancomycin 15 mg/kg IV q8-12h', 'ICMR 2019 Table 3.1 (preferred)'),
 ('Sepsis / septic shock (unclear focus)', 2, 'Meropenem + Vancomycin', 'MEM+VAN', 'Meropenem 1 g IV q8h + Vancomycin 15 mg/kg IV q8-12h', 'ICMR 2019 Table 3.1 (alternative)'),
 ('Sepsis / septic shock (unclear focus)', 2, 'Cefoperazone-sulbactam + Amikacin + Teicoplanin', 'CSL+AMK+TEC', 'Cefop-sulb 3 g IV q12h + Amikacin 15 mg/kg q24h + Teicoplanin 400 mg q12h x3 then daily', 'ICMR 2019 Table 3.1 (alternative)'),
 ('Sepsis / septic shock (unclear focus)', 3, 'Meropenem + Colistin + Vancomycin', 'MEM+COL+VAN', 'Meropenem 1 g q8h + Colistin 9 MU load then 4.5 MU q12h + Vancomycin', 'ICMR 2019 Table 3.1 (CRE risk)'),
 ('Sepsis / septic shock (unclear focus)', 3, 'Meropenem + Polymyxin B + Teicoplanin', 'MEM+PMB+TEC', 'Meropenem 1 g q8h + Polymyxin B 15-20 lakh U load then 7.5-10 lakh U q12h + Teicoplanin', 'ICMR 2019 Table 3.1 (CRE risk)'),
 ('Severe community-acquired pneumonia (ICU)', 1, 'Ceftriaxone + Azithromycin', 'CRO+AZM', 'Ceftriaxone 2 g IV q24h + Azithromycin 500 mg IV q24h', 'ICMR 2019 Ch.4 (CAP)'),
 ('Severe community-acquired pneumonia (ICU)', 2, 'Amoxicillin-clavulanate + Azithromycin', 'AMC+AZM', 'Amox-clav 1.2 g IV q8h + Azithromycin 500 mg q24h', 'ICMR 2019 Ch.4 (CAP)'),
 ('Severe community-acquired pneumonia (ICU)', 3, 'Piperacillin-tazobactam + Azithromycin', 'TZP+AZM', 'Pip-tazo 4.5 g IV q6h + Azithromycin 500 mg q24h', 'ICMR 2019 Ch.4 (Pseudomonas risk)'),
 ('Hospital-acquired / ventilator-associated pneumonia', 1, 'Piperacillin-tazobactam', 'TZP', 'Pip-tazo 4.5 g IV q6h (extended infusion)', 'ICMR 2019 (no MDR risk)'),
 ('Hospital-acquired / ventilator-associated pneumonia', 1, 'Cefoperazone-sulbactam', 'CSL', 'Cefop-sulb 3 g IV q12h', 'ICMR 2019 (no MDR risk)'),
 ('Hospital-acquired / ventilator-associated pneumonia', 2, 'Meropenem', 'MEM', 'Meropenem 1 g IV q8h (extended infusion)', 'ICMR 2019 (MDR risk)'),
 ('Hospital-acquired / ventilator-associated pneumonia', 3, 'Meropenem + Colistin', 'MEM+COL', 'Meropenem 1 g q8h + Colistin 9 MU load then 4.5 MU q12h', 'ICMR 2019 (CR-GNB risk)'),
 ('Hospital-acquired / ventilator-associated pneumonia', 3, 'Meropenem + Colistin + Linezolid', 'MEM+COL+LNZ', 'as above + Linezolid 600 mg IV q12h', 'ICMR 2019 (CR-GNB + MRSA risk)'),
 ('Intra-abdominal infection (secondary peritonitis)', 1, 'Piperacillin-tazobactam', 'TZP', 'Pip-tazo 4.5 g IV q6h', 'ICMR 2019 Sec.2.1.6 (less severe)'),
 ('Intra-abdominal infection (secondary peritonitis)', 1, 'Cefoperazone-sulbactam', 'CSL', 'Cefop-sulb 3 g IV q12h', 'ICMR 2019 Sec.2.1.6 (less severe)'),
 ('Intra-abdominal infection (secondary peritonitis)', 2, 'Ertapenem', 'ETP', 'Ertapenem 1 g IV q24h', 'ICMR 2019 Sec.2.1.6 (severe)'),
 ('Intra-abdominal infection (secondary peritonitis)', 2, 'Meropenem', 'MEM', 'Meropenem 1 g IV q8h', 'ICMR 2019 Sec.2.1.6 (severe)'),
]
pd.DataFrame(REG, columns=['SYNDROME', 'LINE', 'REGIMEN', 'DRUGS', 'DOSING', 'SOURCE']).to_csv(f'{OUT}/icmr_regimens.csv', index=False)

SPEC = {'Acute undifferentiated fever (severe)': 'Blood', 'Sepsis / septic shock (unclear focus)': 'Blood',
        'Severe community-acquired pneumonia (ICU)': 'Resp-CAP', 'Hospital-acquired / ventilator-associated pneumonia': 'Resp-VAP',
        'Intra-abdominal infection (secondary peritonitis)': 'Peritoneal'}
pd.DataFrame(list(SPEC.items()), columns=['SYNDROME', 'SPEC_TYPE']).to_csv(f'{OUT}/specimen_map.csv', index=False)

# ------------------------------------------------------------------ stewardship policy
POL = [
 # code, name, AWaRe, restriction, formulary, culture-evaluable, contraindicated in
 ('AMC', 'Amoxicillin-clavulanate', 'Access', 'free', 1, 1, 'penicillin_anaphylaxis'),
 ('AMK', 'Amikacin', 'Access', 'free', 1, 1, 'aki'),
 ('DOX', 'Doxycycline', 'Access', 'free', 1, 0, 'none'),
 ('CRO', 'Ceftriaxone', 'Watch', 'free', 1, 1, 'none'),
 ('AZM', 'Azithromycin', 'Watch', 'free', 1, 0, 'none'),
 ('TZP', 'Piperacillin-tazobactam', 'Watch', 'justify', 1, 1, 'penicillin_anaphylaxis'),
 ('CSL', 'Cefoperazone-sulbactam', 'Watch', 'justify', 1, 1, 'none'),
 ('ETP', 'Ertapenem', 'Watch', 'justify', 1, 1, 'none'),
 ('IPM', 'Imipenem', 'Watch', 'id_approval', 1, 1, 'none'),
 ('MEM', 'Meropenem', 'Watch', 'id_approval', 1, 1, 'none'),
 ('VAN', 'Vancomycin', 'Watch', 'justify', 1, 1, 'none'),
 ('TEC', 'Teicoplanin', 'Watch', 'justify', 1, 1, 'none'),
 ('COL', 'Colistin', 'Reserve', 'id_approval', 1, 1, 'aki'),
 ('PMB', 'Polymyxin B', 'Reserve', 'id_approval', 1, 1, 'aki'),
 ('LNZ', 'Linezolid', 'Reserve', 'id_approval', 1, 1, 'none'),
]
pd.DataFrame(POL, columns=['CODE', 'NAME', 'AWARE', 'RESTRICTION', 'FORMULARY', 'CULTURE_EVALUABLE', 'CONTRAINDICATED_IN']).to_csv(f'{OUT}/policy.csv', index=False)

# ------------------------------------------------------------------ synthetic WHONET isolates
DRUGS = ['AMC', 'CRO', 'TZP', 'CSL', 'ETP', 'IPM', 'MEM', 'AMK', 'COL', 'PMB', 'VAN', 'TEC', 'LNZ', 'OXA', 'AMP']
GNB = {'Escherichia coli', 'Klebsiella pneumoniae', 'Acinetobacter baumannii', 'Pseudomonas aeruginosa', 'Salmonella Typhi', 'Haemophilus influenzae'}

def phenotype(org, rng, k):
    """Draw one isolate's S/R results with realistic co-resistance. k scales resistance (ICU-B < ICU-A)."""
    r = {d: None for d in DRUGS}
    R = lambda p: 'R' if rng.rand() < min(1, p * k) else 'S'
    if org in ('Escherichia coli', 'Klebsiella pneumoniae'):
        cr_p, esbl_p, amk_p, col_p = (0.25, 0.60, 0.15, 0.02) if org == 'Escherichia coli' else (0.55, 0.30, 0.25, 0.06)
        cr = rng.rand() < cr_p * k; esbl = cr or rng.rand() < esbl_p * min(1, k * 1.2)
        for d in ('IPM', 'MEM', 'ETP'): r[d] = 'R' if cr else 'S'
        r['CRO'] = 'R' if esbl else 'S'; r['AMC'] = 'R' if esbl else R(0.15)
        r['TZP'] = 'R' if cr else (R(0.40) if esbl else R(0.05)); r['CSL'] = 'R' if cr else (R(0.35) if esbl else R(0.05))
        r['AMK'] = R(amk_p + (0.5 if cr else 0)); r['COL'] = R(col_p); r['PMB'] = r['COL']; r['AMP'] = 'R'
    elif org == 'Acinetobacter baumannii':
        cr = rng.rand() < 0.85 * k
        for d in ('IPM', 'MEM'): r[d] = 'R' if cr else 'S'
        for d in ('AMC', 'CRO', 'ETP', 'AMP'): r[d] = 'R'          # intrinsic / unreliable
        r['TZP'] = 'R' if cr else R(0.3); r['CSL'] = R(0.75) if cr else 'S'
        r['AMK'] = R(0.70); r['COL'] = R(0.03); r['PMB'] = r['COL']
    elif org == 'Pseudomonas aeruginosa':
        cr = rng.rand() < 0.35 * k
        for d in ('IPM', 'MEM'): r[d] = 'R' if cr else 'S'
        for d in ('AMC', 'CRO', 'ETP', 'AMP'): r[d] = 'R'          # intrinsic
        r['TZP'] = R(0.8) if cr else R(0.25); r['CSL'] = R(0.7) if cr else R(0.25)
        r['AMK'] = R(0.2 + (0.4 if cr else 0)); r['COL'] = R(0.03); r['PMB'] = r['COL']
    elif org == 'Salmonella Typhi':
        r['CRO'] = R(0.03); r['AMC'] = R(0.2); r['TZP'] = R(0.05); r['CSL'] = R(0.05)
        for d in ('ETP', 'IPM', 'MEM'): r[d] = 'S'
    elif org == 'Haemophilus influenzae':
        r['AMC'] = R(0.10); r['CRO'] = R(0.02); r['TZP'] = R(0.03); r['CSL'] = R(0.03)
        for d in ('ETP', 'IPM', 'MEM'): r[d] = 'S'
    elif org == 'Staphylococcus aureus':
        mrsa = rng.rand() < 0.45 * k; r['OXA'] = 'R' if mrsa else 'S'
        for d in ('AMC', 'CRO', 'TZP', 'CSL', 'ETP', 'IPM', 'MEM'): r[d] = 'R' if mrsa else 'S'
        r['VAN'] = 'S'; r['TEC'] = R(0.01); r['LNZ'] = R(0.01)
    elif org == 'Streptococcus pneumoniae':
        r['CRO'] = R(0.04); r['AMC'] = R(0.05); r['TZP'] = r['AMC']; r['CSL'] = R(0.05)
        for d in ('ETP', 'IPM', 'MEM'): r[d] = R(0.02)
        r['VAN'] = 'S'; r['TEC'] = 'S'; r['LNZ'] = 'S'
    elif org == 'Enterococcus spp.':
        amp = R(0.60); r['AMP'] = amp; r['AMC'] = amp; r['TZP'] = amp; r['IPM'] = amp
        for d in ('CRO', 'ETP', 'MEM', 'CSL'): r[d] = 'R'      # unreliable / intrinsic for enterococci
        r['VAN'] = R(0.08); r['TEC'] = r['VAN']; r['LNZ'] = R(0.01)
    return r

MIX = {  # organism counts per specimen type (ICU A); ICU B uses a scaled-down, less MDR mix
 'Blood': {'Escherichia coli': 80, 'Klebsiella pneumoniae': 90, 'Acinetobacter baumannii': 50, 'Pseudomonas aeruginosa': 35,
           'Staphylococcus aureus': 55, 'Enterococcus spp.': 30, 'Salmonella Typhi': 25},
 'Resp-CAP': {'Streptococcus pneumoniae': 40, 'Haemophilus influenzae': 25, 'Klebsiella pneumoniae': 25,
              'Staphylococcus aureus': 20, 'Pseudomonas aeruginosa': 10},
 'Resp-VAP': {'Klebsiella pneumoniae': 90, 'Acinetobacter baumannii': 110, 'Pseudomonas aeruginosa': 60,
              'Staphylococcus aureus': 25, 'Escherichia coli': 20},
 'Peritoneal': {'Escherichia coli': 70, 'Klebsiella pneumoniae': 45, 'Enterococcus spp.': 25,
                'Pseudomonas aeruginosa': 15, 'Acinetobacter baumannii': 10},
}

def make_icu(name, k, scale, seed, acin_factor=1.0):
    rng = np.random.RandomState(seed); rows = []; pid = 0
    for spec, mix in MIX.items():
        for org, n in mix.items():
            n = int(round(n * scale * (acin_factor if org == 'Acinetobacter baumannii' else 1)))
            for _ in range(n):
                pid += 1; base = dict(ICU=name, PATIENT_ID=f'{name}-{pid:05d}', SPEC_TYPE=spec, ORGANISM=org,
                                      SPEC_DATE=pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(rng.randint(0, 365))))
                ph = phenotype(org, rng, k); rows.append({**base, **ph})
                if rng.rand() < 0.12:   # repeat isolate from same patient -> must be removed by first-isolate rule
                    rows.append({**base, 'SPEC_DATE': base['SPEC_DATE'] + pd.Timedelta(days=3), **phenotype(org, rng, k)})
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{name}_whonet.csv', index=False)
    return df

if __name__ == '__main__':
    a = make_icu('icu_A', k=1.0, scale=1.0, seed=1)
    b = make_icu('icu_B', k=0.35, scale=0.7, seed=2, acin_factor=0.4)
    print('ICU A isolates', len(a), '| ICU B isolates', len(b), '| files in', OUT)
