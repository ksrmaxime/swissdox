# Swissdox – Stance pipeline

Pipeline SLURM (cluster Curnagl, UNIL) qui détecte la critique / l'éloge envers
l'administration publique suisse dans la presse (Swissdox), phrase par phrase,
avec le LLM Apertus-8B-Instruct.

## Étapes

| Étape | sbatch | Script | Sortie (`data/output/…`) |
|---|---|---|---|
| Step 1 – Téléchargement Swissdox | `sbatch sbatch_step1_download.sh` | `scripts/step1_download.py` | `step1_download_job<ID>/swissdox_articles_lead.parquet` |
| Step 2 – Découpage en phrases + filtrage par mots-clés | `sbatch sbatch_step2_sentence_cutting.sh <ID_step1>` | `scripts/step2_sentence_cutting.py` | `step2_sentence_cutting_job<ID>/criticism_base.csv` |
| Step 3a – Classification STANCE (array 8 GPU) | `sbatch sbatch_step3a_stance_array.sh <ID_step2> [MAX_ROWS]` | `scripts/step3a_stance_array.py` | `step3a_stance_array_job<ARRAY_ID>/criticism_stance_task*.parquet` |
| Step 3b – Fusion des résultats de l'array | `sbatch --dependency=afterok:<ID_step3a> sbatch_step3b_stance_merge.sh <ID_step3a>` | (python inline) | `step3b_stance_merge_job<ID>/criticism_stance_merged.{parquet,csv}` |

Chaque étape reçoit en argument le job ID de l'étape précédente.

Labels de stance : `CRITICISM`, `PRAISE`, `NEUTRAL_STATEMENT` — évalués par rapport à
l'entité ciblée (`matched_keywords`) uniquement. Une phrase citant plusieurs entités
distinctes apparaît une fois par entité.

## Code partagé (`src/`)

- `federal_units_table.py` — **liste des mots-clés spécifiques** : copie de
  « Liste des départements fédéraux.csv » (départements + offices, actuels et OLD,
  noms FR/DE, alternatives, abréviations ; colonnes EN pour l'analyse uniquement).
  Pour modifier les mots-clés, c'est ce fichier qu'il faut éditer.
- `download_src.py` — client API Swissdox, construction de la requête, nettoyage du texte,
  mots-clés génériques, expansion des déclinaisons (`Eidgenössische.s.n` → 3 formes),
  variantes d'apostrophe, et `classify_keyword` / `keyword_unit_metadata`.
- `client_src.py` — client LLM (backend `transformers` ou `vllm`).
- `runner_src.py` — application du LLM par batch sur un DataFrame, avec checkpoints.
- `stance_prompts.py` / `stance_config.py` — prompt de stance et sélection des lignes.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .        # rend src/ importable
```

`.env` doit contenir `SWISSDOX_API_KEY` et `SWISSDOX_API_SECRET`.

## Archives

`OLD/` contient les anciennes étapes (populisme, analyse/figures, évaluation GOLD,
pipelines Emb+RoBERTa / comparaison), les notebooks, les anciennes données et logs.
