"""
RAG layer for the Antibiotic Policy Engine: "LLM proposes, rules verify".

1. Knowledge base: three sources are turned into text documents and embedded in one FAISS index
     - ICMR guideline     : chunks of the ICMR 2019 PDF + one document per ICMR regimen row
     - WHONET antibiogram : one document per (ICU, specimen, organism) and per (ICU, syndrome, regimen)
     - Stewardship policy : one document per antibiotic (AWaRe class, restriction, contraindications)
2. Retrieval        : for an ICU case, the top-k documents for that ICU and syndrome are retrieved.
3. Generation       : a local LLM (Ollama) proposes ONE regimen from the ICMR candidate list as JSON,
                      with reasoning and the ids of the documents it relied on.
4. Verification     : the deterministic engine (engine.py) checks the proposal. A contraindicated,
                      non-guideline or resistant choice is overridden; if nothing is reliable the case
                      is escalated to Infectious Disease, whatever the LLM said.
5. Confidence score : 0.5 x agreement with the rule engine + 0.3 x resistance tier + 0.2 x grounding
                      (share of the LLM's cited documents that were actually retrieved).

If Ollama/LangChain are not available, retrieval falls back to TF-IDF and generation falls back to a
template that writes the same JSON from the retrieved documents (marked generator='template').
"""
import json, re
import numpy as np
import pandas as pd
import engine as E

POL_COLS = ['CODE', 'NAME', 'AWARE', 'RESTRICTION', 'FORMULARY', 'CONTRAINDICATED_IN']
FLAG_TEXT = {'aki': 'acute kidney injury', 'penicillin_anaphylaxis': 'penicillin anaphylaxis'}


# ------------------------------------------------------------------ 1. knowledge base
def build_corpus(icus: dict, regimens: pd.DataFrame, policy: pd.DataFrame, spec_map: dict, guideline_chunks=()):
    """Return a list of {'id','text','meta'} documents covering guideline, antibiogram and policy."""
    docs = []
    def add(text, **meta):
        docs.append(dict(id=f'D{len(docs) + 1}', text=text, meta=meta))
    pol = policy.set_index('CODE')
    name = lambda c: pol.loc[c, 'NAME'] if c in pol.index else c

    # ICMR guideline: regimen rows + PDF chunks
    for r in regimens.itertuples():
        add(f"ICMR guideline, {r.SYNDROME}: line {r.LINE} option is {r.REGIMEN}. Dosing: {r.DOSING}. Source: {r.SOURCE}.",
            source='ICMR', syndrome=r.SYNDROME)
    for page, text in guideline_chunks:
        add(f"ICMR guideline text (page {page}): {text.strip()}", source='ICMR-PDF', page=page)

    # Stewardship policy
    for c, p in pol.iterrows():
        restr = {'free': 'unrestricted', 'justify': 'requires prescriber justification',
                 'id_approval': 'requires Infectious Disease approval'}[p.RESTRICTION]
        ci = 'none' if p.CONTRAINDICATED_IN == 'none' else ', '.join(FLAG_TEXT.get(x, x) for x in p.CONTRAINDICATED_IN.split('|'))
        add(f"Hospital stewardship policy: {p.NAME} ({c}) is WHO AWaRe {p.AWARE}, {restr}, "
            f"{'on' if p.FORMULARY == 1 else 'not on'} formulary. Avoid in: {ci}.", source='POLICY', drug=c)

    # WHONET antibiogram, per ICU
    for icu, iso in icus.items():
        ab = E.cumulative_antibiogram(iso)
        for (spec, org), g in ab.groupby(['SPEC_TYPE', 'ORGANISM']):
            n = int(g.N.max())
            parts = ', '.join(f"{name(a)} {v:.0f}%" for a, v in zip(g.ANTIBIOTIC, g.PCT_R) if a in pol.index)
            add(f"WHONET antibiogram {icu}, {spec} isolates, {org} (n={n}): percent resistant - {parts}.",
                source='WHONET', icu=icu, spec=spec)
        for syn, spec in spec_map.items():
            for r in regimens[regimens.SYNDROME == syn].itertuples():
                drugs = [d for d in r.DRUGS.split('+') if pol.loc[d, 'CULTURE_EVALUABLE'] == 1]
                pct, n, per_org = E.regimen_resistance(iso, spec, drugs)
                worst = ', '.join(f"{o} {v}%" for o, v in sorted(per_org.items(), key=lambda x: -x[1])[:3])
                add(f"WHONET coverage {icu}, {syn}: {r.REGIMEN} fails to cover {pct}% of {n} local {spec} isolates "
                    f"(tier {E.TIER_NAMES[E.tier_of(pct)]}). Least covered organisms: {worst}.",
                    source='WHONET', icu=icu, syndrome=syn)
    return docs


# ------------------------------------------------------------------ 2. retrieval
class Retriever:
    """FAISS + Ollama embeddings when available, otherwise TF-IDF. Filters by ICU so another ICU's data is never used."""
    def __init__(self, docs, embeddings=None):
        self.docs = docs; self.mode = 'tfidf'
        if embeddings is not None:
            from langchain_community.vectorstores import FAISS
            self.store = FAISS.from_texts([d['text'] for d in docs], embeddings,
                                          metadatas=[{'i': i} for i in range(len(docs))])
            self.mode = 'faiss'
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit([d['text'] for d in docs])
            self.mat = self.vec.transform([d['text'] for d in docs])

    def _ok(self, d, icu):
        return d['meta'].get('icu') in (None, icu)

    def search(self, query, icu, k=8):
        if self.mode == 'faiss':
            hits = self.store.similarity_search(query, k=k * 4)
            idx = [h.metadata['i'] for h in hits]
        else:
            s = (self.mat @ self.vec.transform([query]).T).toarray().ravel()
            idx = list(s.argsort()[::-1][:k * 4])
        out = [self.docs[i] for i in idx if self._ok(self.docs[i], icu)]
        return out[:k]


def retrieve_case(ret, icu, syndrome, spec, flags, regimens, k_each=4):
    """Pull context from all three sources so the LLM always sees guideline, local resistance and policy."""
    cands = regimens[regimens.SYNDROME == syndrome]
    drugs = sorted({d for r in cands.DRUGS for d in r.split('+')})
    q_icmr = f"ICMR guideline empiric antibiotic regimen for {syndrome}"
    q_whonet = f"WHONET coverage {icu} {syndrome} {spec} isolates resistant regimen"
    q_pol = f"stewardship policy restriction AWaRe {' '.join(drugs)} " + ' '.join(FLAG_TEXT.get(f, f) for f in flags)
    seen, ctx = set(), []
    # structured part of the hybrid retrieval: always include every guideline row and every local coverage
    # summary for this exact ICU + syndrome, so no candidate regimen is missing from the LLM's context
    for d in ret.docs:
        m = d['meta']
        if m.get('syndrome') == syndrome and (m['source'] == 'ICMR' or (m['source'] == 'WHONET' and m.get('icu') == icu)):
            seen.add(d['id']); ctx.append(d)
    q_whonet = f"WHONET antibiogram {icu} {spec} isolates percent resistant"
    # semantic part: guideline text, organism-level antibiogram and policy documents
    for q, src in ((q_icmr, ('ICMR-PDF',)), (q_whonet, ('WHONET',)), (q_pol, ('POLICY',))):
        added = 0
        for d in ret.search(q, icu, k=k_each * 6):
            if d['meta']['source'] in src and d['id'] not in seen and 'fails to cover' not in d['text']:
                seen.add(d['id']); ctx.append(d); added += 1
                if added >= k_each: break
    return ctx


# ------------------------------------------------------------------ 3. generation
PROMPT = """You are an antimicrobial stewardship assistant for an Indian ICU.
Choose ONE empiric regimen for the case below, using ONLY the context documents.
Rules:
- Pick from the CANDIDATE REGIMENS list exactly as written, or answer "ESCALATE" if every candidate is more than 30% resistant locally.
- Do not use any drug the patient must avoid. Prefer lower local resistance, then narrower AWaRe class (Access < Watch < Reserve).
- Never invent drugs, doses or numbers that are not in the context.
- Cite the document ids (like D12) you relied on.
Return JSON only: {{"regimen": "...", "reasoning": "2-4 sentences", "citations": ["D..", "D.."]}}

CASE: {icu}; syndrome: {syndrome}; culture specimen: {spec}; patient factors: {flags}
CANDIDATE REGIMENS: {cands}

CONTEXT:
{context}
"""

def make_prompt(case, ctx, candidates):
    return PROMPT.format(icu=case['icu'], syndrome=case['syndrome'], spec=case['spec'],
                         flags=', '.join(FLAG_TEXT.get(f, f) for f in case['flags']) or 'none',
                         cands=' | '.join(candidates),
                         context='\n'.join(f"[{d['id']}] {d['text']}" for d in ctx))

def parse_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

def template_generate(case, ctx, candidates):
    """No-LLM fallback: choose the candidate whose WHONET coverage document shows the lowest resistance."""
    best, best_pct, cite = None, 1e9, []
    for d in ctx:
        if d['meta']['source'] == 'WHONET' and 'fails to cover' in d['text']:
            for c in candidates:
                if f": {c} fails" in d['text']:
                    pct = float(re.search(r'cover ([\d.]+)%', d['text']).group(1))
                    if pct < best_pct: best, best_pct, cite = c, pct, [d['id']]
    if best is None or best_pct > E.CAUTION_MAX:
        return dict(regimen='ESCALATE', reasoning='No candidate regimen in the retrieved local data is below 30% resistance.', citations=cite)
    return dict(regimen=best, reasoning=f'{best} has the lowest local resistance ({best_pct}%) among retrieved candidates.', citations=cite)


# ------------------------------------------------------------------ 4-5. verification + confidence
def verify(proposal, engine_res, ctx):
    ranked = engine_res['ranked']; by_name = {c['regimen']: c for c in ranked}
    excluded = {c['regimen'] for c in engine_res['excluded']}
    eng_action = engine_res['decision']['action']; eng_best = engine_res['decision'].get('regimen')
    choice = (proposal.get('regimen') or '').strip()
    ctx_ids = {d['id'] for d in ctx}; cites = [c for c in proposal.get('citations', []) if isinstance(c, str)]
    grounding = (sum(c in ctx_ids for c in cites) / len(cites)) if cites else 0.0

    if eng_action == 'ESCALATE':
        status = 'AGREE' if choice.upper() == 'ESCALATE' else 'OVERRIDDEN'
        final, reason = 'ESCALATE', ('' if status == 'AGREE' else
                                      f"LLM proposed '{choice}', but every guideline regimen is >30% resistant in this ICU.")
        agreement = tier_score = (1.0 if status == 'AGREE' else 0.0)
    elif choice in excluded:
        status, final, agreement, tier_score = 'OVERRIDDEN', eng_best, 0.0, 0.0
        reason = f"'{choice}' is contraindicated for this patient."
    elif choice not in by_name:
        status, final, agreement, tier_score = 'OVERRIDDEN', eng_best, 0.0, 0.0
        reason = f"'{choice}' is not an ICMR candidate for this syndrome."
    else:
        c = by_name[choice]
        if c['tier'] == 3:
            status, final, agreement, tier_score = 'OVERRIDDEN', eng_best, 0.0, 0.0
            reason = f"'{choice}' is {c['pct_resistant']}% resistant locally (no reliable option)."
        else:
            status = 'AGREE' if choice == eng_best else 'ACCEPTED (alternative)'
            final, reason = choice, ''
            agreement = 1.0 if choice == eng_best else 0.6
            tier_score = 1.0 if c['tier'] == 1 else 0.5
    conf = round(0.5 * agreement + 0.3 * tier_score + 0.2 * grounding, 2)
    level = 'HIGH' if conf >= 0.8 else ('MEDIUM' if conf >= 0.5 else 'LOW')
    return dict(final=final, status=status, override_reason=reason, confidence=conf, level=level,
                grounding=round(grounding, 2))


def run_case(case, icus, regimens, policy, spec_map, ret, llm=None):
    iso = icus[case['icu']]
    eng = E.recommend(case['syndrome'], iso, regimens, policy, spec_map, patient_flags=case['flags'])
    case = dict(case, spec=spec_map[case['syndrome']])
    candidates = [c['regimen'] for c in eng['ranked']]            # contraindicated regimens already removed
    ctx = retrieve_case(ret, case['icu'], case['syndrome'], case['spec'], case['flags'], regimens)
    if llm is not None:
        raw = llm(make_prompt(case, ctx, candidates)); proposal = parse_json(raw); gen = 'llm'
    else:
        proposal = template_generate(case, ctx, candidates); gen = 'template'
    v = verify(proposal, eng, ctx)
    return dict(icu=case['icu'], syndrome=case['syndrome'], flags=case['flags'], generator=gen,
                retrieval=ret.mode, proposal=proposal, engine_decision=eng['decision'], **v,
                context=[{'id': d['id'], 'source': d['meta']['source'], 'text': d['text'][:300]} for d in ctx])
