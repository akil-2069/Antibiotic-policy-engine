# Antibiotic Policy Engine for Indian ICUs

**Live demo:** https://akil-2069.github.io/Antibiotic-policy-engine/?v=2
**Try yourself now by uploading any icu report given in "Try your own icu" box** 

## The problem

When a patient in an Indian ICU becomes septic, the doctor usually has to start antibiotics before culture results come back. That first choice is called *empiric therapy*, and it is often taken straight from a national guideline such as ICMR's.

National guidelines can't know what is resistant in *your* ICU. A regimen that works well in one hospital can fail in another, where most *Klebsiella* already resist carbapenems. That information exists in every microbiology lab's WHONET database, but it rarely reaches the bedside in a usable form.

## What this project does

It combines three sources that usually sit in different places and gives one recommendation for a given syndrome:

| Source | What it answers | File |
|---|---|---|
| **ICMR Treatment Guidelines (2019)** | Which regimens are allowed for this syndrome? | `data/icmr_regimens.csv` |
| **The ICU's WHONET export** | Which of those regimens still work *here*? | `data/icu_A_whonet.csv`, `data/icu_B_whonet.csv` |
| **The hospital's stewardship policy** | Which working regimen should we try first? | `data/policy.csv` |

For a chosen syndrome, the engine:

1. Takes every regimen ICMR lists for it.
2. Maps the syndrome to the matching culture type, for example ventilator pneumonia to respiratory aspirates.
3. Keeps only the first isolate per patient (the CLSI M39 rule), then measures what share of local isolates each regimen fails to cover. An isolate counts as covered if it is susceptible to at least one drug in the regimen.
4. Sorts each regimen into a tier: **Safe** (≤10% resistant), **Use with caution** (10–30%), or **No reliable option** (>30%).
5. Applies the hospital's rules: removes drugs contraindicated for this patient (e.g. colistin in kidney injury) and flags drugs that need Infectious Disease approval.
6. Ranks what's left: safest tier first, then the narrowest WHO AWaRe class, the least restricted drug, and the fewest drugs.
7. If every option is above 30%, it **does not guess**. It tells the team to send cultures and call Microbiology or Infectious Disease.

### How AI is used: RAG + LLM, verified by rules

On top of the rule engine sits a retrieval-augmented generation (RAG) layer, `rag_engine.py`:

1. **Knowledge base:** all three sources become text documents in one **FAISS** vector store (embeddings: Ollama `nomic-embed-text`): chunks of the ICMR guideline PDF and its regimen table, the WHONET antibiogram written as per-ICU summaries (per organism and per regimen), and the stewardship policy (one entry per drug).
2. **Hybrid retrieval:** for each ICU case, every guideline regimen and local coverage summary for that ICU and syndrome, plus the closest guideline text, organism antibiograms and policy entries.
3. **Generation:** a local LLM (**Ollama, qwen2.5:3b**, run through LangChain) proposes one regimen from the ICMR candidates as JSON, with its reasoning and the ids of the documents it used.
4. **Verification:** the deterministic engine checks the proposal. A contraindicated, non-guideline or >30%-resistant choice is **overridden**, and a case with no reliable option is always escalated, whatever the LLM said.
5. **Confidence score** (for the LLM's proposal): `0.5 × agreement with the rule engine + 0.3 × resistance tier of its choice + 0.2 × grounding`, where grounding is the share of the LLM's citations that were really retrieved. Reported as HIGH (≥ 0.8), MEDIUM (≥ 0.5) or LOW. An overridden proposal always scores LOW.

This keeps what LLMs are good at (reading evidence and explaining a choice) while making sure a hallucinated drug or a locally resistant regimen never reaches the final answer.

## Results

Same guideline and same policy, two different ICUs:

| Syndrome | ICU A (tertiary, high resistance) | ICU B (secondary, lower resistance) |
|---|---|---|
| Sepsis / septic shock | Meropenem + Colistin + Vancomycin · Safe, 2.2% · *needs ID approval* | Cefoperazone-sulbactam + Amikacin + Teicoplanin · Safe, 2.1% |
| Ventilator-associated pneumonia | Meropenem + Colistin · Safe, 4.9% · *needs ID approval* | Meropenem + Colistin · Safe, 1.8% |
| Severe community pneumonia | **Escalate**: every option is >30% resistant | Pip-tazo + Azithromycin · Safe, 8.2% |
| Intra-abdominal infection | **Escalate**: every option is >30% resistant | Pip-tazo · Use with caution, 20.7% |
| Severe undifferentiated fever | **Escalate**: every option is >30% resistant | Pip-tazo + Doxycycline · Use with caution, 18.8% |

In ICU B, the engine reaches for colistin only where it's actually needed. For sepsis it picks a Watch-class combination, because that is already safe there. In ICU A, the same syndromes push it toward Reserve drugs or a specialist call.

The same sepsis patient in ICU A **with acute kidney injury** loses every colistin, polymyxin and amikacin option. What remains is over 30% resistant, so the engine escalates instead of quietly recommending a regimen that would likely fail.

![Antibiograms](figures/fig_antibiograms.png)

### RAG + LLM evaluation

30 cases (2 ICUs × 5 syndromes × 3 patient profiles: none, kidney injury, penicillin allergy) run through FAISS retrieval → Ollama qwen2.5:3b → rule verification on Google Colab. Full results: `results/rag_results.csv`, summary: `results/rag_summary.json`.

| Metric | Value |
|---|---|
| Valid JSON answers from the LLM | 30 / 30 (100%) |
| Citations that were really retrieved (grounding) | 100% |
| LLM chose the same regimen as the rule engine | 12 / 30 (40%) |
| LLM chose a different but still safe regimen (accepted) | 3 / 30 (10%) |
| **Unsafe LLM choices caught and overridden by the rule engine** | **15 / 30 (50%)** |
| Confidence levels | 13 HIGH · 2 MEDIUM · 15 LOW |
| Average time per case | 6.7 s |

**What this shows:** the split between the two ICUs is the important part.

- **ICU B (lower resistance):** the LLM's choice was accepted in 14 of 15 cases.
- **ICU A (high resistance):** the LLM's choice was overridden in 14 of 15 cases. It kept picking familiar first-line guideline options (for example ceftriaxone + azithromycin for community pneumonia, or meropenem + vancomycin for sepsis) even though the retrieved antibiogram showed them to be more than 30% resistant in that ICU. The rule engine replaced them with the locally active regimen, or escalated to Infectious Disease when nothing was reliable.

In other words, a small LLM reading the right evidence still defaults to "textbook" answers. That is exactly the failure that matters most in high-resistance ICUs, and it is why every LLM proposal here is verified before it is shown.

### Validation

`results/validation_tests.csv`: **11 of 11 checks pass**, including:

- tier boundaries at exactly 10% and 30%
- repeat isolates from the same patient are removed
- restricted drugs carry the ID-approval flag
- escalation triggers when nothing is reliable
- the kidney-injury flag removes nephrotoxic options
- the same input gives the same audit fingerprint
- different ICU data gives different recommendations

## Run it

**In the browser:** open the demo link. Pick an ICU and a syndrome, tick patient factors, and click any regimen to see which organisms it misses. You can also upload your own WHONET-style CSV, and it never leaves your browser.

**In Colab:** upload `Antibiotic_Policy_Engine.ipynb` → **Runtime → Change runtime type → T4 GPU** → **Runtime → Run all**. It takes about 10 minutes, mostly the one-time Ollama model download. If Ollama or the ICMR PDF download fails, it falls back to TF-IDF retrieval and a template generator and still finishes, and every result records which path was used.

**LLM results in the web demo:** `docs/rag_results.json` holds the 30 LLM runs from the notebook. The demo shows them in an "AI recommendation · RAG + LLM" panel for ICU A and ICU B: what the LLM proposed, whether the rule engine agreed or overrode it, and the confidence score. To refresh them, re-run the notebook and replace that file with `outputs/rag_results.json`.

**Locally:**
```bash
pip install pandas numpy matplotlib scikit-learn
python make_data.py
python -c "import pandas as pd, engine as E; \
reg=pd.read_csv('data/icmr_regimens.csv'); pol=pd.read_csv('data/policy.csv'); \
sm=dict(pd.read_csv('data/specimen_map.csv').values); iso=pd.read_csv('data/icu_A_whonet.csv'); \
print(E.pretty(E.recommend('Sepsis / septic shock (unclear focus)', iso, reg, pol, sm)))"
```

**For a real ICU:** replace `data/icu_A_whonet.csv` with the unit's WHONET export (same column names: `PATIENT_ID`, `SPEC_TYPE`, `ORGANISM`, `SPEC_DATE`, one column per drug code with S/I/R) and edit `data/policy.csv` to match the hospital's restricted list. Nothing else changes.

## Turn on the demo link

1. Push this repo to GitHub as a **public** repository.
2. Go to **Settings → Pages**, set **Source** to *Deploy from a branch*, choose **main** and the **/docs** folder, and click **Save**.
3. After a minute your demo is live at `https://<your-username>.github.io/antibiotic-policy-engine/`. Put that link at the top of this README and in applications.

## Project layout

```
engine.py                       the rule engine (deterministic, ~150 lines)
rag_engine.py                   RAG + LLM layer: knowledge base, hybrid retrieval, generation, rule verification, confidence
make_data.py                    builds the regimen table, policy and synthetic WHONET exports
Antibiotic_Policy_Engine.ipynb  end-to-end notebook: data, engine, heatmaps, tests, RAG + LLM, Gradio UI
data/                           inputs (replace with a real ICU's files)
results/                        recommendations, antibiograms, validation tests, RAG + LLM results
docs/index.html                 the browser demo (served by GitHub Pages)
docs/rag_results.json           LLM outputs exported by the notebook, shown in the demo
```

## Limitations

- **Synthetic antibiogram data.** The two ICUs are synthetic, with resistance rates set to roughly the ranges published by ICMR's AMR surveillance network. They show how the engine behaves; they don't describe any real hospital.
- **Curated regimen table.** The table covers five syndromes and was curated by hand from the 2019 ICMR guideline. It should be checked against the current edition before any real use.
- **No dosing adjustment.** Doses are shown as written for normal kidney function. The engine doesn't adjust for weight, renal function or drug interactions.
- **Small local LLM.** qwen2.5:3b can misread context or return invalid JSON; that is why every proposal goes through the rule engine before it is shown.
- **Not for clinical use.** This is a student project, built to show how guideline, surveillance and policy data can be joined. It should not be used for patient care.

## References

- ICMR, *Treatment Guidelines for Antimicrobial Use in Common Syndromes*, 2nd ed., 2019. [PDF](https://www.icmr.gov.in/icmrobject/custom_data/pdf/resource-guidelines/Treatment_Guidelines_2019_Final.pdf)
- ICMR, *Antimicrobial Resistance Research and Surveillance Network Annual Report 2024*. [PDF](https://www.icmr.gov.in/icmrobject/uploads/Report/1763981012_icmramrsnannualreport2024.pdf)
- CLSI M39, *Analysis and Presentation of Cumulative Antimicrobial Susceptibility Test Data*.
- WHO AWaRe classification of antibiotics.
- WHONET microbiology laboratory database software, WHO Collaborating Centre for Surveillance of Antimicrobial Resistance.

## Author

**Akilan Murali**, B.Tech Computer Science and Engineering, VIT Vellore
