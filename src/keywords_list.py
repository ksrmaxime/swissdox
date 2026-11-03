# keywords_list.py
"""
LISTE COMPLÈTE DES MOTS-CLÉS — source unique utilisée par le code.

Tout est écrit en toutes lettres, déclinaisons allemandes comprises : ce qui
figure ici est exactement ce qui est envoyé à Swissdox (step 1) et cherché dans
les phrases (step 2). Pour ajouter / retirer un mot-clé, modifier ce fichier.

Seule transformation automatique : chaque terme contenant une apostrophe est
aussi cherché avec l'autre apostrophe (' <-> ’).

Origine : « Liste des départements fédéraux.csv » (oct. 2026) + correction des
terminaisons allemandes par une collègue germanophone (nov. 2026). Le commentaire
au-dessus de chaque nom allemand rappelle la notation d'origine (ex. "Bundesamt.s.es").
Règle d'accord appliquée : le génitif du nom (-s/-es) n'apparaît qu'avec la forme
en -n de l'adjectif ("des Eidgenössischen Departements").
"""

# ════════════════════════════════════════════════════════════════════════════
# MOTS-CLÉS GÉNÉRIQUES
# forme de base : (sub_type, [toutes les formes cherchées])
# ════════════════════════════════════════════════════════════════════════════
GENERIC_KEYWORDS = {
    # öffentliche.n.r Verwaltung.en
    'öffentliche Verwaltung': ('Public Admin', [
        'öffentliche Verwaltung', 'öffentlichen Verwaltung', 'öffentlicher Verwaltung',
        'öffentliche Verwaltungen', 'öffentlichen Verwaltungen', 'öffentlicher Verwaltungen',
    ]),
    'Bundesverwaltung': ('Public Admin', ['Bundesverwaltung', 'Bundesverwaltungen']),  # Bundesverwaltung.en
    'Staatsverwaltung': ('Public Admin', ['Staatsverwaltung', 'Staatsverwaltungen']),  # Staatsverwaltung.en
    'Bürokratie': ('bureaucracy', ['Bürokratie']),
    'Bürokraten': ('bureaucracy', ['Bürokraten']),
    'Bürokratinnen': ('bureaucracy', ['Bürokratinnen']),
    'Beamte': ('Public Admin', ['Beamte', 'Beamten']),  # Beamte.n
    'Administration publique': ('Public Admin', ['Administration publique']),
    'Administration fédérale': ('Public Admin', ['Administration fédérale']),
    "Administration d'état": ('Public Admin', ["Administration d'état"]),
    'Bureaucratie': ('bureaucracy', ['Bureaucratie']),
    'bureaucrates': ('bureaucracy', ['bureaucrates']),
    'fonctionnaires': ('Public Admin', ['fonctionnaires']),
}

# ════════════════════════════════════════════════════════════════════════════
# MOTS-CLÉS SPÉCIFIQUES : UNITÉS FÉDÉRALES (une entrée = une unité)
#   type / upper_level / en_* : métadonnées pour l'analyse (EN jamais cherché)
#   fr_names    : noms français (principal + alternatives)
#   de_names    : noms allemands ; chaque sous-liste = un nom avec TOUTES ses
#                 déclinaisons, la 1re forme étant la forme de référence
#   *_abbrev    : abréviation principale (sert de keyword_abbrev)
#   *_alt_abbrevs : abréviations alternatives
# ════════════════════════════════════════════════════════════════════════════
FEDERAL_UNITS = [
    # ──────────────── FDFA ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDFA',
        "en_name": 'Federal Department of Foreign Affairs', "en_abbrev": 'FDFA',
        "fr_names": ['Département fédéral des affaires étrangères'],
        "fr_abbrev": 'DFAE', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s für auswärtige Angelegenheiten
            [
                'Eidgenössische Departement für auswärtige Angelegenheiten',
                'Eidgenössisches Departement für auswärtige Angelegenheiten',
                'Eidgenössischen Departement für auswärtige Angelegenheiten',
                'Eidgenössischen Departements für auswärtige Angelegenheiten',
            ],
        ],
        "de_abbrev": 'EDA', "de_alt_abbrevs": [],
    },
    # ──────────────── FDHA ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDHA',
        "en_name": 'Federal Department of Home Affairs', "en_abbrev": 'FDHA',
        "fr_names": ["Département fédéral de l'intérieur"],
        "fr_abbrev": 'DFI', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s des Innern
            [
                'Eidgenössische Departement des Innern',
                'Eidgenössisches Departement des Innern',
                'Eidgenössischen Departement des Innern',
                'Eidgenössischen Departements des Innern',
            ],
        ],
        "de_abbrev": 'EDI', "de_alt_abbrevs": [],
    },
    # ──────────────── FDJP ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDJP',
        "en_name": 'Federal Department of Justice and Police', "en_abbrev": 'FDJP',
        "fr_names": ['Département fédéral de justice et police'],
        "fr_abbrev": 'DFJP', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Justiz- und Polizeidepartement.s
            [
                'Eidgenössische Justiz- und Polizeidepartement',
                'Eidgenössisches Justiz- und Polizeidepartement',
                'Eidgenössischen Justiz- und Polizeidepartement',
                'Eidgenössischen Justiz- und Polizeidepartements',
            ],
        ],
        "de_abbrev": 'EJPD', "de_alt_abbrevs": [],
    },
    # ──────────────── DDPS ────────────────
    {
        "type": 'Federal Department', "upper_level": 'DDPS',
        "en_name": 'Federal Department of Defence, Civil Protection and Sport', "en_abbrev": 'DDPS',
        "fr_names": ['Département fédéral de la défense, de la protection de la population et des sports'],
        "fr_abbrev": 'DDPS', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s für Verteidigung, Bevölkerungsschutz und Sport
            [
                'Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport',
                'Eidgenössisches Departement für Verteidigung, Bevölkerungsschutz und Sport',
                'Eidgenössischen Departement für Verteidigung, Bevölkerungsschutz und Sport',
                'Eidgenössischen Departements für Verteidigung, Bevölkerungsschutz und Sport',
            ],
        ],
        "de_abbrev": 'VBS', "de_alt_abbrevs": [],
    },
    # ──────────────── FDF ────────────────
    {
        "type": 'Federal Department', "upper_level": 'FDF',
        "en_name": 'Federal Department of Finance', "en_abbrev": 'FDF',
        "fr_names": ['Département fédéral des finances'],
        "fr_abbrev": 'DFF', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Finanzdepartement.s
            [
                'Eidgenössische Finanzdepartement',
                'Eidgenössisches Finanzdepartement',
                'Eidgenössischen Finanzdepartement',
                'Eidgenössischen Finanzdepartements',
            ],
        ],
        "de_abbrev": 'EFD', "de_alt_abbrevs": [],
    },
    # ──────────────── EAER ────────────────
    {
        "type": 'Federal Department', "upper_level": 'EAER',
        "en_name": 'Federal Department of Economic Affairs, Education and Research', "en_abbrev": 'EAER',
        "fr_names": ["Département fédéral de l'économie, de la formation et de la recherche", 'Département fédéral de l’économie'],
        "fr_abbrev": 'DEFR', "fr_alt_abbrevs": ['DFE'],
        "de_names": [
            # Eidgenössische.s.n Departement.s für Wirtschaft, Bildung und Forschung
            [
                'Eidgenössische Departement für Wirtschaft, Bildung und Forschung',
                'Eidgenössisches Departement für Wirtschaft, Bildung und Forschung',
                'Eidgenössischen Departement für Wirtschaft, Bildung und Forschung',
                'Eidgenössischen Departements für Wirtschaft, Bildung und Forschung',
            ],
            # Eidgenössische.s.n Volkswirtschaftsdepartement.s
            [
                'Eidgenössische Volkswirtschaftsdepartement',
                'Eidgenössisches Volkswirtschaftsdepartement',
                'Eidgenössischen Volkswirtschaftsdepartement',
                'Eidgenössischen Volkswirtschaftsdepartements',
            ],
        ],
        "de_abbrev": 'WBF', "de_alt_abbrevs": ['EVD'],
    },
    # ──────────────── DETEC ────────────────
    {
        "type": 'Federal Department', "upper_level": 'DETEC',
        "en_name": 'Federal Department of the Environment, Transport, Energy and Communications', "en_abbrev": 'DETEC',
        "fr_names": ["Département fédéral de l'environnement, des transports, de l'énergie et de la communication"],
        "fr_abbrev": 'DETEC', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Departement.s für Umwelt, Verkehr, Energie und Kommunikation
            [
                'Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation',
                'Eidgenössisches Departement für Umwelt, Verkehr, Energie und Kommunikation',
                'Eidgenössischen Departement für Umwelt, Verkehr, Energie und Kommunikation',
                'Eidgenössischen Departements für Umwelt, Verkehr, Energie und Kommunikation',
            ],
        ],
        "de_abbrev": 'UVEK', "de_alt_abbrevs": [],
    },
    # ──────────────── FDFA ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDFA',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DFAE', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EDA', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'FDFA State Secretariat', "en_abbrev": 'FDFA-STS',
        "fr_names": ['Secrétariat d’Etat du DFAE', 'Secrétariat d’Etat'],
        "fr_abbrev": 'SEE-DFAE', "fr_alt_abbrevs": ['SEE'],
        "de_names": [
            # Staatssekretariat.s EDA
            [
                'Staatssekretariat EDA',
                'Staatssekretariats EDA',
            ],
            # Staatssekretariat.s
            [
                'Staatssekretariat',
                'Staatssekretariats',
            ],
        ],
        "de_abbrev": 'STS-EDA', "de_alt_abbrevs": ['STS'],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Directorate of International Law', "en_abbrev": 'DIL',
        "fr_names": ['Direction du droit international public'],
        "fr_abbrev": 'DDIP', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für Völkerrecht'],
        ],
        "de_abbrev": 'DV', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Swiss Agency for Development and Cooperation', "en_abbrev": 'SDC',
        "fr_names": ['Direction du développement et de la coopération'],
        "fr_abbrev": 'DDC', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für Entwicklung und Zusammenarbeit'],
        ],
        "de_abbrev": 'DEZA', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Directorate for Resources', "en_abbrev": 'DR',
        "fr_names": ['Direction des ressources', 'Direction des ressources et du réseau extérieur'],
        "fr_abbrev": 'DR', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für Ressourcen'],
            ['Direktion für Ressourcen und Aussennetz'],
        ],
        "de_abbrev": 'DR', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDFA',
        "en_name": 'Consular Directorate', "en_abbrev": 'CD',
        "fr_names": ['Direction consulaire'],
        "fr_abbrev": 'DC', "fr_alt_abbrevs": [],
        "de_names": [
            # Konsularische.n Direktion
            [
                'Konsularische Direktion',
                'Konsularischen Direktion',
            ],
        ],
        "de_abbrev": 'KD', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDFA',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Direction politique'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Politische.n Direktion
            [
                'Politische Direktion',
                'Politischen Direktion',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDFA',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Direction des affaires européennes'],
        "fr_abbrev": 'DAE', "fr_alt_abbrevs": [],
        "de_names": [
            ['Direktion für europäische Angelegenheiten'],
        ],
        "de_abbrev": 'DEA', "de_alt_abbrevs": [],
    },
    # ──────────────── FDHA ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDHA',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DFI', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EDI', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office for Gender Equality', "en_abbrev": 'FOGE',
        "fr_names": ['Bureau fédéral de l’égalité entre femmes et hommes'],
        "fr_abbrev": 'BFEG', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Büro.s für die Gleichstellung von Frau und Mann
            [
                'Eidgenössische Büro für die Gleichstellung von Frau und Mann',
                'Eidgenössisches Büro für die Gleichstellung von Frau und Mann',
                'Eidgenössischen Büro für die Gleichstellung von Frau und Mann',
                'Eidgenössischen Büros für die Gleichstellung von Frau und Mann',
            ],
        ],
        "de_abbrev": 'EBG', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office of Culture', "en_abbrev": 'FOC',
        "fr_names": ['Office fédéral de la culture'],
        "fr_abbrev": 'OFC', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Kultur
            [
                'Bundesamt für Kultur',
                'Bundesamts für Kultur',
                'Bundesamtes für Kultur',
            ],
        ],
        "de_abbrev": 'BAK', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Swiss Federal Archives', "en_abbrev": 'SFA',
        "fr_names": ['Archives fédérales suisses'],
        "fr_abbrev": 'AFS', "fr_alt_abbrevs": [],
        "de_names": [
            # Schweizerische.s.n Bundesarchiv.s
            [
                'Schweizerische Bundesarchiv',
                'Schweizerisches Bundesarchiv',
                'Schweizerischen Bundesarchiv',
                'Schweizerischen Bundesarchivs',
            ],
        ],
        "de_abbrev": 'BAR', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office of Meteorology and Climatology', "en_abbrev": 'MeteoSwiss',
        "fr_names": ['Office fédéral de météorologie et de climatologie'],
        "fr_abbrev": 'MétéoSuisse', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Meteorologie und Klimatologie
            [
                'Bundesamt für Meteorologie und Klimatologie',
                'Bundesamts für Meteorologie und Klimatologie',
                'Bundesamtes für Meteorologie und Klimatologie',
            ],
        ],
        "de_abbrev": 'MeteoSchweiz', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Office of Public Health', "en_abbrev": 'FOPH',
        "fr_names": ['Office fédéral de la santé publique'],
        "fr_abbrev": 'OFSP', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Gesundheit
            [
                'Bundesamt für Gesundheit',
                'Bundesamts für Gesundheit',
                'Bundesamtes für Gesundheit',
            ],
        ],
        "de_abbrev": 'BAG', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Statistical Office', "en_abbrev": 'FSO',
        "fr_names": ['Office fédéral de la statistique'],
        "fr_abbrev": 'OFS', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Statistik
            [
                'Bundesamt für Statistik',
                'Bundesamts für Statistik',
                'Bundesamtes für Statistik',
            ],
        ],
        "de_abbrev": 'BFS', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Social Insurance Office', "en_abbrev": 'FSIO',
        "fr_names": ['Office fédéral des assurances sociales'],
        "fr_abbrev": 'OFAS', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Sozialversicherungen
            [
                'Bundesamt für Sozialversicherungen',
                'Bundesamts für Sozialversicherungen',
                'Bundesamtes für Sozialversicherungen',
            ],
        ],
        "de_abbrev": 'BSV', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDHA',
        "en_name": 'Federal Food Safety and Veterinary Office', "en_abbrev": 'FSVO',
        "fr_names": ['Office fédéral de la sécurité alimentaire et des affaires vétérinaires'],
        "fr_abbrev": 'OSAV', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Lebensmittelsicherheit und Veterinärwesen
            [
                'Bundesamt für Lebensmittelsicherheit und Veterinärwesen',
                'Bundesamts für Lebensmittelsicherheit und Veterinärwesen',
                'Bundesamtes für Lebensmittelsicherheit und Veterinärwesen',
            ],
        ],
        "de_abbrev": 'BLV', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDHA',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Office fédéral de l’assurance militaire'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Militärversicherung
            [
                'Bundesamt für Militärversicherung',
                'Bundesamts für Militärversicherung',
                'Bundesamtes für Militärversicherung',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDHA',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Groupement de la science et de la recherche', 'Secrétariat d’Etat à l’éducation et à la recherche'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Gruppe für Wissenschaft und Forschung'],
            # Staatssekretariat.s für Bildung und Forschung
            [
                'Staatssekretariat für Bildung und Forschung',
                'Staatssekretariats für Bildung und Forschung',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    # ──────────────── FDJP ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDJP',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DFJP', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EJPD', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'Federal office of Justice', "en_abbrev": 'FOJ',
        "fr_names": ['Office fédéral de la justice'],
        "fr_abbrev": 'OFJ', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Justiz
            [
                'Bundesamt für Justiz',
                'Bundesamts für Justiz',
                'Bundesamtes für Justiz',
            ],
        ],
        "de_abbrev": 'BJ', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'Federal Office of Police', "en_abbrev": 'Fedpol',
        "fr_names": ['Office fédéral de la police'],
        "fr_abbrev": 'Fedpol', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Polizei
            [
                'Bundesamt für Polizei',
                'Bundesamts für Polizei',
                'Bundesamtes für Polizei',
            ],
        ],
        "de_abbrev": 'Fedpol', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDJP',
        "en_name": 'State Secretariat for  Migration', "en_abbrev": 'SEM',
        "fr_names": ["Secrétariat d'état aux migrations", 'Office fédéral des étrangers', 'Office fédéral des réfugiés', 'Office fédéral des migrations'],
        "fr_abbrev": 'SEM', "fr_alt_abbrevs": ['ODM'],
        "de_names": [
            # Staatssekretariat.s für Migration
            [
                'Staatssekretariat für Migration',
                'Staatssekretariats für Migration',
            ],
            # Bundesamt.s.es für Ausländerfragen
            [
                'Bundesamt für Ausländerfragen',
                'Bundesamts für Ausländerfragen',
                'Bundesamtes für Ausländerfragen',
            ],
            # Bundesamt.s.es für Flüchtlinge
            [
                'Bundesamt für Flüchtlinge',
                'Bundesamts für Flüchtlinge',
                'Bundesamtes für Flüchtlinge',
            ],
            # Bundesamt.s.es für Migration
            [
                'Bundesamt für Migration',
                'Bundesamts für Migration',
                'Bundesamtes für Migration',
            ],
        ],
        "de_abbrev": 'SEM', "de_alt_abbrevs": ['BFM'],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDJP',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Office fédéral des assurances privées'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Privatversicherungen
            [
                'Bundesamt für Privatversicherungen',
                'Bundesamts für Privatversicherungen',
                'Bundesamtes für Privatversicherungen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDJP',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Office fédéral de métrologie'],
        "fr_abbrev": 'METAS', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Institut.s für Metrologie
            [
                'Eidgenössische Institut für Metrologie',
                'Eidgenössisches Institut für Metrologie',
                'Eidgenössischen Institut für Metrologie',
                'Eidgenössischen Instituts für Metrologie',
            ],
            # Eidgenössische.s.n Amt.s.es für Messwesen
            [
                'Eidgenössische Amt für Messwesen',
                'Eidgenössisches Amt für Messwesen',
                'Eidgenössischen Amt für Messwesen',
                'Eidgenössischen Amts für Messwesen',
                'Eidgenössischen Amtes für Messwesen',
            ],
        ],
        "de_abbrev": 'METAS', "de_alt_abbrevs": [],
    },
    # ──────────────── DDPS ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-DDPS',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DDPS', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-VBS', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'State Secretariat for Security Policy', "en_abbrev": 'SEPOS',
        "fr_names": ["Secrétariat d'Etat à la politique de sécurité", 'Direction de la politique de sécurité'],
        "fr_abbrev": 'SEPOS', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für Sicherheitspolitik
            [
                'Staatssekretariat für Sicherheitspolitik',
                'Staatssekretariats für Sicherheitspolitik',
            ],
            ['Direktion für Sicherheitspolitik'],
        ],
        "de_abbrev": 'SEPOS', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Intelligence Service', "en_abbrev": 'FIS',
        "fr_names": ['Service de renseignement de la Confédération', 'Direction du Renseignement stratégique'],
        "fr_abbrev": 'SRC', "fr_alt_abbrevs": [],
        "de_names": [
            # Nachrichtendienst.es des Bundes
            [
                'Nachrichtendienst des Bundes',
                'Nachrichtendienstes des Bundes',
            ],
            # Direktion für strategische.n.r Nachrichtendienst
            [
                'Direktion für strategische Nachrichtendienst',
                'Direktion für strategischen Nachrichtendienst',
                'Direktion für strategischer Nachrichtendienst',
            ],
        ],
        "de_abbrev": 'NDB', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Office of the Armed Force Attorney General', "en_abbrev": 'OAFAG',
        "fr_names": ['Office de l’auditeur en chef'],
        "fr_abbrev": 'OAC', "fr_alt_abbrevs": [],
        "de_names": [
            # Oberauditorat.s
            [
                'Oberauditorat',
                'Oberauditorats',
            ],
        ],
        "de_abbrev": 'OA', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Defence Group', "en_abbrev": '',
        "fr_names": ['Groupement Défense'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Gruppe Verteidigung'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office for Defence Procurement', "en_abbrev": 'armasuisse',
        "fr_names": ['Office fédéral de l’armement', 'Groupement de l’armement', 'Groupement armasuisse'],
        "fr_abbrev": 'armasuisse', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Rüstung
            [
                'Bundesamt für Rüstung',
                'Bundesamts für Rüstung',
                'Bundesamtes für Rüstung',
            ],
            ['Gruppe Rüstung'],
            ['Gruppe armasuisse'],
        ],
        "de_abbrev": 'armasuisse', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office of Topography', "en_abbrev": 'swisstopo',
        "fr_names": ['Office fédéral de topographie'],
        "fr_abbrev": 'swisstopo', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Landestopografie
            [
                'Bundesamt für Landestopografie',
                'Bundesamts für Landestopografie',
                'Bundesamtes für Landestopografie',
            ],
        ],
        "de_abbrev": 'swisstopo', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office for Civil Protection', "en_abbrev": 'FOCP',
        "fr_names": ['Office fédéral de la protection de la population', 'Office fédéral de la protection civile'],
        "fr_abbrev": 'OFPP', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Bevölkerungsschutz
            [
                'Bundesamt für Bevölkerungsschutz',
                'Bundesamts für Bevölkerungsschutz',
                'Bundesamtes für Bevölkerungsschutz',
            ],
            # Bundesamt.s.es für Zivilschutz
            [
                'Bundesamt für Zivilschutz',
                'Bundesamts für Zivilschutz',
                'Bundesamtes für Zivilschutz',
            ],
        ],
        "de_abbrev": 'BABS', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'Federal Office of Sport', "en_abbrev": 'FOSPO',
        "fr_names": ['Office fédéral du sport'],
        "fr_abbrev": 'OFSPO', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Sport
            [
                'Bundesamt für Sport',
                'Bundesamts für Sport',
                'Bundesamtes für Sport',
            ],
        ],
        "de_abbrev": 'BASPO', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DDPS',
        "en_name": 'National Cyber Security Centre', "en_abbrev": 'NCSC',
        "fr_names": ['Office fédéral de la cybersécurité'],
        "fr_abbrev": 'OFCS', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Cybersicherheit
            [
                'Bundesamt für Cybersicherheit',
                'Bundesamts für Cybersicherheit',
                'Bundesamtes für Cybersicherheit',
            ],
        ],
        "de_abbrev": 'BACS', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DDPS',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Forces aériennes'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            ['Luftwaffe'],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DDPS',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Forces terrestres'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Heer.es
            [
                'Heer',
                'Heeres',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DDPS',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Etat-major général'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalstab.s.es
            [
                'Generalstab',
                'Generalstabs',
                'Generalstabes',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    # ──────────────── FDF ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-FDF',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DFF', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-EFD', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'State Secretariat for International Finance', "en_abbrev": 'SIF',
        "fr_names": ["Secrétariat d'état aux questions financières internationales"],
        "fr_abbrev": 'SFI', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für internationale Finanzfragen
            [
                'Staatssekretariat für internationale Finanzfragen',
                'Staatssekretariats für internationale Finanzfragen',
            ],
        ],
        "de_abbrev": 'SIF', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Finance Administration', "en_abbrev": 'FFA',
        "fr_names": ['Administration fédérale des finances'],
        "fr_abbrev": 'AFF', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n.r Finanzverwaltung
            [
                'Eidgenössische Finanzverwaltung',
                'Eidgenössischen Finanzverwaltung',
                'Eidgenössischer Finanzverwaltung',
            ],
        ],
        "de_abbrev": 'EFV', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office of Personnel', "en_abbrev": 'FOPER',
        "fr_names": ['Office fédéral du personnel'],
        "fr_abbrev": 'OFPER', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.s.n Personalamt.s.es
            [
                'Eidgenössische Personalamt',
                'Eidgenössisches Personalamt',
                'Eidgenössischen Personalamt',
                'Eidgenössischen Personalamts',
                'Eidgenössischen Personalamtes',
            ],
        ],
        "de_abbrev": 'EPA', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Tax Administration', "en_abbrev": 'FTA',
        "fr_names": ['Administration fédérale des contributions'],
        "fr_abbrev": 'AFC', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n.r Steuerverwaltung
            [
                'Eidgenössische Steuerverwaltung',
                'Eidgenössischen Steuerverwaltung',
                'Eidgenössischer Steuerverwaltung',
            ],
        ],
        "de_abbrev": 'ESTV', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office for Customs and Border Security', "en_abbrev": 'FOCBS',
        "fr_names": ['Office fédéral de la douane et de la sécurité des frontières', 'Office fédéral des douanes'],
        "fr_abbrev": 'OFDF', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Zoll und Grenzsicherheit
            [
                'Bundesamt für Zoll und Grenzsicherheit',
                'Bundesamts für Zoll und Grenzsicherheit',
                'Bundesamtes für Zoll und Grenzsicherheit',
            ],
            # Eidgenössische.n.r Zollverwaltung
            [
                'Eidgenössische Zollverwaltung',
                'Eidgenössischen Zollverwaltung',
                'Eidgenössischer Zollverwaltung',
            ],
        ],
        "de_abbrev": 'BAZG', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office of Information Technology, Systems and Telecommunication', "en_abbrev": 'FOITT',
        "fr_names": ['Office fédéral de l’informatique et de la télécommunication', 'Office fédéral de l’informatique'],
        "fr_abbrev": 'OFIT', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Informatik und Telekommunikation
            [
                'Bundesamt für Informatik und Telekommunikation',
                'Bundesamts für Informatik und Telekommunikation',
                'Bundesamtes für Informatik und Telekommunikation',
            ],
            # Bundesamt.s.es für Informatik
            [
                'Bundesamt für Informatik',
                'Bundesamts für Informatik',
                'Bundesamtes für Informatik',
            ],
        ],
        "de_abbrev": 'BIT', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'FDF',
        "en_name": 'Federal Office for Buildings and Logistics', "en_abbrev": 'FOBL',
        "fr_names": ['Office fédéral des constructions et de la logistique'],
        "fr_abbrev": 'OFCL', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Bauten und Logistik
            [
                'Bundesamt für Bauten und Logistik',
                'Bundesamts für Bauten und Logistik',
                'Bundesamtes für Bauten und Logistik',
            ],
        ],
        "de_abbrev": 'BBL', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDF',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Caisse fédérale d’assurance', 'Office fédéral des assurances privées'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Eidgenössische.n.r Versicherungskasse
            [
                'Eidgenössische Versicherungskasse',
                'Eidgenössischen Versicherungskasse',
                'Eidgenössischer Versicherungskasse',
            ],
            # Bundesamt.s.es für Privatversicherungen
            [
                'Bundesamt für Privatversicherungen',
                'Bundesamts für Privatversicherungen',
                'Bundesamtes für Privatversicherungen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'FDF',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Unité de pilotage informatique de la Confédération'],
        "fr_abbrev": 'UPIC', "fr_alt_abbrevs": [],
        "de_names": [
            # Informatiksteuerungsorgan.s des Bundes
            [
                'Informatiksteuerungsorgan des Bundes',
                'Informatiksteuerungsorgans des Bundes',
            ],
        ],
        "de_abbrev": 'ISB', "de_alt_abbrevs": [],
    },
    # ──────────────── EAER ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-EAER',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DEFR', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-WBF', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Office of the Price Supervisor', "en_abbrev": 'PRS',
        "fr_names": ['Surveillance des prix'],
        "fr_abbrev": 'SPR', "fr_alt_abbrevs": [],
        "de_names": [
            ['Preisüberwachung'],
        ],
        "de_abbrev": 'PUE', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'State Secretariat for Economic Affairs', "en_abbrev": 'SECO',
        "fr_names": ['Secrétariat d’Etat à l’économie'],
        "fr_abbrev": 'SECO', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für Wirtschaft
            [
                'Staatssekretariat für Wirtschaft',
                'Staatssekretariats für Wirtschaft',
            ],
        ],
        "de_abbrev": 'SECO', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'State Secretariat for Education, Research and Innovation', "en_abbrev": 'SERI',
        "fr_names": ["Secrétariat d'Etat à la formation, à la recherche et à l’innovation", 'Office fédéral de la formation professionnelle et de la technologie'],
        "fr_abbrev": 'SEFRI', "fr_alt_abbrevs": [],
        "de_names": [
            # Staatssekretariat.s für Bildung, Forschung und Innovation
            [
                'Staatssekretariat für Bildung, Forschung und Innovation',
                'Staatssekretariats für Bildung, Forschung und Innovation',
            ],
            # Bundesamt.s.es für Berufsbildung und Technologie
            [
                'Bundesamt für Berufsbildung und Technologie',
                'Bundesamts für Berufsbildung und Technologie',
                'Bundesamtes für Berufsbildung und Technologie',
            ],
        ],
        "de_abbrev": 'SBFI', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for Agriculture', "en_abbrev": 'FOAG',
        "fr_names": ['Office fédéral de l’agriculture'],
        "fr_abbrev": 'OFAG', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Landwirtschaft
            [
                'Bundesamt für Landwirtschaft',
                'Bundesamts für Landwirtschaft',
                'Bundesamtes für Landwirtschaft',
            ],
        ],
        "de_abbrev": 'BLW', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for National Economic Supply', "en_abbrev": 'FONES',
        "fr_names": ['Office fédéral pour l’approvisionnement économique du pays'],
        "fr_abbrev": 'OFAE', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für wirtschaftliche Landesversorgung
            [
                'Bundesamt für wirtschaftliche Landesversorgung',
                'Bundesamts für wirtschaftliche Landesversorgung',
                'Bundesamtes für wirtschaftliche Landesversorgung',
            ],
        ],
        "de_abbrev": 'BWL', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for Housing', "en_abbrev": 'FOH',
        "fr_names": ['Office fédéral du logement'],
        "fr_abbrev": 'OFL', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Wohnungswesen
            [
                'Bundesamt für Wohnungswesen',
                'Bundesamts für Wohnungswesen',
                'Bundesamtes für Wohnungswesen',
            ],
        ],
        "de_abbrev": 'BWO', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'EAER',
        "en_name": 'Federal Office for Civilian Service', "en_abbrev": 'CIVI',
        "fr_names": ['Office fédéral du service civil'],
        "fr_abbrev": 'CIVI', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Zivildienst
            [
                'Bundesamt für Zivildienst',
                'Bundesamts für Zivildienst',
                'Bundesamtes für Zivildienst',
            ],
        ],
        "de_abbrev": 'ZIVI', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'EAER',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Office vétérinaire fédéral'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Veterinärwesen
            [
                'Bundesamt für Veterinärwesen',
                'Bundesamts für Veterinärwesen',
                'Bundesamtes für Veterinärwesen',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
    # ──────────────── DETEC ────────────────
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'General Secretariat', "en_abbrev": 'GS-DETEC',
        "fr_names": ['Secrétariat général'],
        "fr_abbrev": 'SG-DETEC', "fr_alt_abbrevs": [],
        "de_names": [
            # Generalsekretariat.s
            [
                'Generalsekretariat',
                'Generalsekretariats',
            ],
        ],
        "de_abbrev": 'GS-UVEK', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office of Transport', "en_abbrev": 'FOT',
        "fr_names": ['Office fédéral des transports'],
        "fr_abbrev": 'OFT', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Verkehr
            [
                'Bundesamt für Verkehr',
                'Bundesamts für Verkehr',
                'Bundesamtes für Verkehr',
            ],
        ],
        "de_abbrev": 'BAV', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office of Civil Aviation', "en_abbrev": 'FOCA',
        "fr_names": ['Office fédéral de l’aviation civile'],
        "fr_abbrev": 'OFAC', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Zivilluftfahrt
            [
                'Bundesamt für Zivilluftfahrt',
                'Bundesamts für Zivilluftfahrt',
                'Bundesamtes für Zivilluftfahrt',
            ],
        ],
        "de_abbrev": 'BAZL', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Swiss Federal Office of Energy', "en_abbrev": 'SFOE',
        "fr_names": ['Office fédéral de l’énergie'],
        "fr_abbrev": 'OFEN', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Energie
            [
                'Bundesamt für Energie',
                'Bundesamts für Energie',
                'Bundesamtes für Energie',
            ],
        ],
        "de_abbrev": 'BFE', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Roads Office', "en_abbrev": 'FEDRO',
        "fr_names": ['Office fédéral des routes'],
        "fr_abbrev": 'OFROU', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Strassen
            [
                'Bundesamt für Strassen',
                'Bundesamts für Strassen',
                'Bundesamtes für Strassen',
            ],
        ],
        "de_abbrev": 'ASTRA', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office of Communications', "en_abbrev": 'OFCOM',
        "fr_names": ['Office fédéral de la communication'],
        "fr_abbrev": 'OFCOM', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Kommunikation
            [
                'Bundesamt für Kommunikation',
                'Bundesamts für Kommunikation',
                'Bundesamtes für Kommunikation',
            ],
        ],
        "de_abbrev": 'BAKOM', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office for the Environment', "en_abbrev": 'FOEN',
        "fr_names": ['Office fédéral de l’environnement', 'Office fédéral de l’environnement, des forêts et du paysage'],
        "fr_abbrev": 'OFEV', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Umwelt
            [
                'Bundesamt für Umwelt',
                'Bundesamts für Umwelt',
                'Bundesamtes für Umwelt',
            ],
            # Bundesamt.s.es für Umwelt, Wald und Landschaft
            [
                'Bundesamt für Umwelt, Wald und Landschaft',
                'Bundesamts für Umwelt, Wald und Landschaft',
                'Bundesamtes für Umwelt, Wald und Landschaft',
            ],
        ],
        "de_abbrev": 'BAFU', "de_alt_abbrevs": [],
    },
    {
        "type": 'Federal Offices', "upper_level": 'DETEC',
        "en_name": 'Federal Office for Spatial Development', "en_abbrev": 'ARE',
        "fr_names": ['Office fédéral du développement territorial'],
        "fr_abbrev": 'ARE', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Raumentwicklung
            [
                'Bundesamt für Raumentwicklung',
                'Bundesamts für Raumentwicklung',
                'Bundesamtes für Raumentwicklung',
            ],
        ],
        "de_abbrev": 'ARE', "de_alt_abbrevs": [],
    },
    {
        "type": 'OLD Federal Offices', "upper_level": 'DETEC',
        "en_name": '', "en_abbrev": '',
        "fr_names": ['Office fédéral des eaux et de la géologie'],
        "fr_abbrev": '', "fr_alt_abbrevs": [],
        "de_names": [
            # Bundesamt.s.es für Wasser und Geologie
            [
                'Bundesamt für Wasser und Geologie',
                'Bundesamts für Wasser und Geologie',
                'Bundesamtes für Wasser und Geologie',
            ],
        ],
        "de_abbrev": '', "de_alt_abbrevs": [],
    },
]
