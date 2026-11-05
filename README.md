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

### Grande période : téléchargement année par année (recommandé)

Une requête Swissdox sur plusieurs années peut rester bloquée côté serveur.
Pour une longue période, utiliser l'enchaînement annuel :

```bash
sbatch sbatch_step1_download_yearly.sh 2000 2025
```

- un job par année, chacun soumet automatiquement l'année suivante : les
  requêtes Swissdox sont strictement consécutives (jamais en parallèle) ;
- résultats par année dans `data/output/step1_download_years_chain<ID>/year_YYYY/` ;
- après la dernière année, `sbatch_step1b_merge_years.sh` est soumis
  automatiquement et produit `data/output/step1_download_job<ID merge>/swissdox_articles_lead.parquet`
  (+ `articles_per_year.csv`) ;
- puis : `sbatch sbatch_step2_sentence_cutting.sh <ID merge>` (16 CPU en parallèle) ;
- reprise après échec d'une année : `sbatch sbatch_step1_download_yearly.sh <année> 2025 <dossier de la chaîne>`
  (les années déjà téléchargées sont sautées).

Labels de stance : `CRITICISM`, `PRAISE`, `NEUTRAL_STATEMENT` — évalués par rapport à
l'entité ciblée (`matched_keywords`) uniquement. Une phrase citant plusieurs entités
distinctes apparaît une fois par entité.

## Code partagé (`src/`)

- `keywords_list.py` — **liste complète des mots-clés, écrite en toutes lettres**
  (génériques + départements et offices fédéraux, actuels et OLD ; noms FR/DE avec
  toutes les déclinaisons allemandes, alternatives, abréviations ; colonnes EN pour
  l'analyse uniquement). C'est la seule source : pour modifier les mots-clés,
  éditer ce fichier.
- `download_src.py` — client API Swissdox, construction de la requête, nettoyage du texte,
  variantes d'apostrophe (' / ’), et `classify_keyword` / `keyword_unit_metadata`.
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
