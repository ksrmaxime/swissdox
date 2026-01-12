from __future__ import annotations

import os
from pathlib import Path
import argparse
import pandas as pd
from dotenv import load_dotenv

from swissdox.pipelines.emb_roberta import EmbRobertaPipelineConfig, run_emb_roberta_pipeline
from swissdox.embeddings import EmbeddingThemeConfig
from swissdox.sentences import SentenceSplitConfig
from swissdox.sentiment import SentimentConfig

# (Pour l’instant on laisse ici; ensuite on pourra les déplacer dans swissdox/config.py)
THEMES = [
    ("Foreign Affairs", "FR: diplomatie, relations internationales, commerce extérieur, coopération, ambassade, consulat, ONU/UE. DE: Diplomatie, Aussenbeziehungen, Aussenhandel, Zusammenarbeit, Botschaft, Konsulat, UNO/EU."),
    ("Culture", "FR: culture, art, musique, théâtre, musée, patrimoine. DE: Kultur, Kunst, Musik, Theater, Museum, Kulturerbe."),
    ("Health", "FR: santé, médecins, hôpitaux, soins, LAMal, primes, pandémie. DE: Gesundheit, Ärzte, Spitäler, Pflege, KVG, Prämien, Pandemie."),
    ("Social", "FR: affaires sociales, personnes âgées, AVS/AI, pauvreté, aides sociales. DE: Sozialwesen, Senioren, AHV/IV, Armut, Sozialhilfe."),
    ("Justice", "FR: droit, tribunaux, police, criminalité, justice, surveillance. DE: Recht, Gerichte, Polizei, Kriminalität, Justiz, Überwachung."),
    ("Migration", "FR: asile, migration, immigration, réfugiés, permis, étrangers, SEM. DE: Asyl, Migration, Einwanderung, Flüchtlinge, Bewilligungen, Ausländer, SEM."),
    ("Defence", "FR: défense, armée, sécurité, protection civile, cyberattaque. DE: Verteidigung, Armee, Sicherheit, Bevölkerungsschutz, Cyberangriff."),
    ("Sport", "FR: sport, clubs, fédérations, compétitions, promotion du sport. DE: Sport, Vereine, Verbände, Wettkämpfe, Sportförderung."),
    ("Finance", "FR: finances publiques, budget, impôts, TVA, fiscalité. DE: Finanzen, Budget, Steuern, MWST, Fiskalpolitik."),
    ("Economy", "FR: économie, entreprises, banques, commerce, marché du travail. DE: Wirtschaft, Unternehmen, Banken, Handel, Arbeitsmarkt."),
    ("Education", "FR: école, université, collège, formation, enseignants, élèves. DE: Schule, Universität, Gymnasium, Bildung, Lehrpersonen, Schüler."),
    ("Research", "FR: recherche, innovation, science, laboratoires, technologie. DE: Forschung, Innovation, Wissenschaft, Labore, Technologie."),
    ("Environment", "FR: écologie, climat, CO2, biodiversité, protection de la nature. DE: Umwelt, Klima, CO2, Biodiversität, Naturschutz."),
    ("Transports", "FR: transports, routes, mobilité, CFF, trains, voitures, avions. DE: Verkehr, Strassen, Mobilität, SBB, Züge, Autos, Flugzeuge."),
    ("Energy", "FR: énergie, nucléaire, électricité, gaz, pétrole, charbon. DE: Energie, Atomkraft, Strom, Gas, Öl, Kohle."),
    ("Communication", "FR: communication, médias, TV, radio, internet, réseaux, antennes. DE: Kommunikation, Medien, Fernsehen, Radio, Internet, Netze, Antennen."),
]

KW_SENT = [
    "Bürokratie","Berner Verwaltung","Papierkrieg","Verwaltung","Bundesverwaltung",
    "Beamtenapparat","Amtsschimmel","Regulierungsdichte","Behörden","Bürokraten",
    "Beamte","Staatsangestellte",
    "Bureaucratie","Administration publique","Administration fédérale","Appareil administratif",
    "Appareil étatique","Appareil de l’État","Autorités administratives","Services de l’État",
    "Services publics","Fonction publique","Pouvoir administratif","Autorités cantonales",
    "Administration centrale","Départements fédéraux","Offices fédéraux","Organes de l’État",
    "Technocratie","Bureaucrates","Fonctionnaires","Employés de l'État",
    "VBS","DDPS","Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport",
    "Département fédéral de la défense, de la protection de la population et des sports",
    "EDA","DFAE","Eidgenössische Departement für auswärtige Angelegenheiten",
    "Département fédéral des affaires étrangères",
    "UVEK","DETEC","Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation",
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication",
    "EJPD","DFJP","Eidgenössische Justiz- und Polizeidepartement",
    "Département fédéral de justice et police",
    "EDI","DFI","Eidgenössische Departement des Innern","Département fédéral de l'intérieur",
    "EFD","DFF","Eidgenössische Finanzdepartement","Département fédéral des finances",
    "WBF","DEFR","Eidgenössische Departement für Wirtschaft, Bildung und Forschung",
    "Département fédéral de l'économie, de la formation et de la recherche",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/swissdox_articles_raw.parquet")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run scripts/download_articles.py first.")

    df = pd.read_parquet(in_path)

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    cfg = EmbRobertaPipelineConfig(
        themes_cfg=EmbeddingThemeConfig(batch_size=64, threshold=0.25, other_label="Others", max_content_chars_fallback=1200),
        split_cfg=SentenceSplitConfig(),
        sentiment_cfg=SentimentConfig(batch_size=64, max_length=256, text_col="sentence"),
    )

    df_themed, df_sent, df_final = run_emb_roberta_pipeline(
        df,
        themes=THEMES,
        keywords=KW_SENT,
        cfg=cfg,
        hf_token=hf_token,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "swissdox_articles_with_theme.parquet"
    p2 = out_dir / "swissdox_sentences_filtered.parquet"
    p3 = out_dir / "swissdox_sentences_with_sentiment.parquet"
    p4 = out_dir / "swissdox_phrases_theme_sentiment.csv"

    df_themed.to_parquet(p1, index=False)
    df_sent.to_parquet(p2, index=False)
    df_final.to_parquet(p3, index=False)
    df_final.to_csv(p4, index=False, encoding="utf-8-sig")

    print("✅ Saved:", p1)
    print("✅ Saved:", p2)
    print("✅ Saved:", p3)
    print("✅ Saved:", p4)


if __name__ == "__main__":
    main()
