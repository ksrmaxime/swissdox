# federal_units_table.py
"""
Table des unités fédérales utilisée pour les mots-clés spécifiques.

Copie conforme de "Liste des départements fédéraux.csv" (RESULTS_SWISSDOX, oct. 2026),
une ligne par unité. Pour modifier la liste : modifier ce tableau (ou régénérer depuis le CSV).

Conventions (interprétées dans src/download_src.py) :
  - "Eidgenössische.s.n" = déclinaisons : Eidgenössische / Eidgenössisches / Eidgenössischen
  - " / " dans une cellule sépare plusieurs noms alternatifs
  - TYPE commençant par "OLD" = ancienne unité, utilisée comme les autres
  - Colonnes EN : uniquement pour les analyses, jamais utilisées comme mots-clés

Correction par rapport au CSV : "awisstopo" (abréviation FR, faute de frappe) -> "swisstopo".
"""

FIELDS = (
    'type'          ,  # TYPE
    'upper_level'   ,  # UPPER LEVEL
    'nom_fr'        ,  # NOM FR
    'alt_nom_fr'    ,  # Alternative NOM FR
    'abbr_fr'       ,  # Abbréviation FR
    'alt_abbr_fr'   ,  # Alternative Abbr FR
    'nom_de'        ,  # NOM DE
    'alt_nom_de'    ,  # Alternative NOM DE
    'abbr_de'       ,  # Abbréviation DE
    'alt_abbr_de'   ,  # Alternative Abbr DE
    'nom_en'        ,  # NOM EN (Paper Use Only)
    'abbr_en'       ,  # Abbréviation EN (Paper Use Only)
)

FEDERAL_UNITS_RAW = [
    ('Federal Department', 'FDFA', 'Département fédéral des affaires étrangères', '', 'DFAE', '', 'Eidgenössische.s.n Departement.s für auswärtige Angelegenheiten', '', 'EDA', '', 'Federal Department of Foreign Affairs', 'FDFA'),
    ('Federal Department', 'FDHA', "Département fédéral de l'intérieur", '', 'DFI', '', 'Eidgenössische.s.n Departement.s des Innern', '', 'EDI', '', 'Federal Department of Home Affairs', 'FDHA'),
    ('Federal Department', 'FDJP', 'Département fédéral de justice et police', '', 'DFJP', '', 'Eidgenössische.s.n Justiz- und Polizeidepartement.s', '', 'EJPD', '', 'Federal Department of Justice and Police', 'FDJP'),
    ('Federal Department', 'DDPS', 'Département fédéral de la défense, de la protection de la population et des sports', '', 'DDPS', '', 'Eidgenössische.s.n Departement.s für Verteidigung, Bevölkerungsschutz und Sport', '', 'VBS', '', 'Federal Department of Defence, Civil Protection and Sport', 'DDPS'),
    ('Federal Department', 'FDF', 'Département fédéral des finances', '', 'DFF', '', 'Eidgenössische.s.n Finanzdepartement.s', '', 'EFD', '', 'Federal Department of Finance', 'FDF'),
    ('Federal Department', 'EAER', "Département fédéral de l'économie, de la formation et de la recherche", 'Département fédéral de l’économie', 'DEFR', 'DFE', 'Eidgenössische.s.n Departement.s für Wirtschaft, Bildung und Forschung', 'Eidgenössische.s.n Volkswirtschaftsdepartement', 'WBF', 'EVD', 'Federal Department of Economic Affairs, Education and Research', 'EAER'),
    ('Federal Department', 'DETEC', "Département fédéral de l'environnement, des transports, de l'énergie et de la communication", '', 'DETEC', '', 'Eidgenössische.s.n Departement.s für Umwelt, Verkehr, Energie und Kommunikation', '', 'UVEK', '', 'Federal Department of the Environment, Transport, Energy and Communications', 'DETEC'),
    ('Federal Offices', 'FDFA', 'Secrétariat général', '', 'SG-DFAE', '', 'Generalsekretariat', '', 'GS-EDA', '', 'General Secretariat', 'GS-FDFA'),
    ('Federal Offices', 'FDFA', 'Secrétariat d’Etat du DFAE', 'Secrétariat d’Etat', 'SEE-DFAE', 'SEE', 'Staatssekretariat EDA', 'Staatssekretariat', 'STS-EDA', 'STS', 'FDFA State Secretariat', 'FDFA-STS'),
    ('Federal Offices', 'FDFA', 'Direction du droit international public', '', 'DDIP', '', 'Direktion für Völkerrecht', '', 'DV', '', 'Directorate of International Law', 'DIL'),
    ('Federal Offices', 'FDFA', 'Direction du développement et de la coopération', '', 'DDC', '', 'Direktion für Entwicklung und Zusammenarbeit', '', 'DEZA', '', 'Swiss Agency for Development and Cooperation', 'SDC'),
    ('Federal Offices', 'FDFA', 'Direction des ressources', 'Direction des ressources et du réseau extérieur', 'DR', '', 'Direktion für Ressourcen', 'Direktion für Ressourcen und Aussennetz', 'DR', '', 'Directorate for Resources', 'DR'),
    ('Federal Offices', 'FDFA', 'Direction consulaire', '', 'DC', '', 'Konsularische.n.r Direktion', '', 'KD', '', 'Consular Directorate', 'CD'),
    ('OLD Federal Offices', 'FDFA', 'Direction politique', '', '', '', 'Politische.n.r Direktion', '', '', '', '', ''),
    ('OLD Federal Offices', 'FDFA', 'Direction des affaires européennes', '', 'DAE', '', 'Direktion für europäische Angelegenheiten', '', 'DEA', '', '', ''),
    ('Federal Offices', 'FDHA', 'Secrétariat général', '', 'SG-DFI', '', 'Generalsekretariat', '', 'GS-EDI', '', 'General Secretariat', 'GS-FDHA'),
    ('Federal Offices', 'FDHA', 'Bureau fédéral de l’égalité entre femmes et hommes', '', 'BFEG', '', 'Eidgenössische.s.n Büro für die Gleichstellung von Frau und Mann', '', 'EBG', '', 'Federal Office for Gender Equality', 'FOGE'),
    ('Federal Offices', 'FDHA', 'Office fédéral de la culture', '', 'OFC', '', 'Bundesamt für Kultur', '', 'BAK', '', 'Federal Office of Culture', 'FOC'),
    ('Federal Offices', 'FDHA', 'Archives fédérales suisses', '', 'AFS', '', 'Schweizerische.s.n Bundesarchiv', '', 'BAR', '', 'Swiss Federal Archives', 'SFA'),
    ('Federal Offices', 'FDHA', 'Office fédéral de météorologie et de climatologie', '', 'MétéoSuisse', '', 'Bundesamt für Meteorologie und Klimatologie', '', 'MeteoSchweiz', '', 'Federal Office of Meteorology and Climatology', 'MeteoSwiss'),
    ('Federal Offices', 'FDHA', 'Office fédéral de la santé publique', '', 'OFSP', '', 'Bundesamt für Gesundheit', '', 'BAG', '', 'Federal Office of Public Health', 'FOPH'),
    ('Federal Offices', 'FDHA', 'Office fédéral de la statistique', '', 'OFS', '', 'Bundesamt für Statistik', '', 'BFS', '', 'Federal Statistical Office', 'FSO'),
    ('Federal Offices', 'FDHA', 'Office fédéral des assurances sociales', '', 'OFAS', '', 'Bundesamt für Sozialversicherungen', '', 'BSV', '', 'Federal Social Insurance Office', 'FSIO'),
    ('Federal Offices', 'FDHA', 'Office fédéral de la sécurité alimentaire et des affaires vétérinaires', '', 'OSAV', '', 'Bundesamt für Lebensmittelsicherheit und Veterinärwesen', '', 'BLV', '', 'Federal Food Safety and Veterinary Office', 'FSVO'),
    ('OLD Federal Offices', 'FDHA', 'Office fédéral de l’assurance militaire', '', '', '', 'Bundesamt für Militärversicherung', '', '', '', '', ''),
    ('OLD Federal Offices', 'FDHA', 'Groupement de la science et de la recherche', 'Secrétariat d’Etat à l’éducation et à la recherche', '', '', 'Gruppe für Wissenschaft und Forschung', 'Staatssekretariat für Bildung und Forschung', '', '', '', ''),
    ('Federal Offices', 'FDJP', 'Secrétariat général', '', 'SG-DFJP', '', 'Generalsekretariat', '', 'GS-EJPD', '', 'General Secretariat', 'GS-FDJP'),
    ('Federal Offices', 'FDJP', 'Office fédéral de la justice', '', 'OFJ', '', 'Bundesamt für Justiz', '', 'BJ', '', 'Federal office of Justice', 'FOJ'),
    ('Federal Offices', 'FDJP', 'Office fédéral de la police', '', 'Fedpol', '', 'Bundesamt für Polizei', '', 'Fedpol', '', 'Federal Office of Police', 'Fedpol'),
    ('Federal Offices', 'FDJP', "Secrétariat d'état aux migrations", 'Office fédéral des étrangers / Office fédéral des réfugiés / Office fédéral des migrations', 'SEM', 'ODM', 'Staatssekretariat für Migration', 'Bundesamt für Ausländerfragen / Bundesamt für Flüchtlinge / Bundesamt für Migration', 'SEM', 'BFM', 'State Secretariat for  Migration', 'SEM'),
    ('OLD Federal Offices', 'FDJP', 'Office fédéral des assurances privées', '', '', '', 'Bundesamt für Privatversicherungen', '', '', '', '', ''),
    ('OLD Federal Offices', 'FDJP', 'Office fédéral de métrologie', '', 'METAS', '', 'Eidgenössische.s.n  Institut für Metrologie', 'Eidgenössische Amt für Messwesen', 'METAS', '', '', ''),
    ('Federal Offices', 'DDPS', 'Secrétariat général', '', 'SG-DDPS', '', 'Generalsekretariat', '', 'GS-VBS', '', 'General Secretariat', 'GS-DDPS'),
    ('Federal Offices', 'DDPS', "Secrétariat d'Etat à la politique de sécurité", 'Direction de la politique de sécurité', 'SEPOS', '', 'Staatssekretariat für Sicherheitspolitik', 'Direktion für Sicherheitspolitik', 'SEPOS', '', 'State Secretariat for Security Policy', 'SEPOS'),
    ('Federal Offices', 'DDPS', 'Service de renseignement de la Confédération', 'Direction du Renseignement stratégique', 'SRC', '', 'Nachrichtendienst des Bundes', 'Direktion für strategische.n.r Nachrichtendienst', 'NDB', '', 'Federal Intelligence Service', 'FIS'),
    ('Federal Offices', 'DDPS', 'Office de l’auditeur en chef', '', 'OAC', '', 'Oberauditorat', '', 'OA', '', 'Office of the Armed Force Attorney General', 'OAFAG'),
    ('Federal Offices', 'DDPS', 'Groupement Défense', '', '', '', 'Gruppe Verteidigung', '', '', '', 'Defence Group', ''),
    ('Federal Offices', 'DDPS', 'Office fédéral de l’armement', 'Groupement de l’armement / Groupement armasuisse', 'armasuisse', '', 'Bundesamt für Rüstung', 'Gruppe Rüstung / Gruppe armasuisse', 'armasuisse', '', 'Federal Office for Defence Procurement', 'armasuisse'),
    ('Federal Offices', 'DDPS', 'Office fédéral de topographie', '', 'swisstopo', '', 'Bundesamt für Landestopografie', '', 'swisstopo', '', 'Federal Office of Topography', 'swisstopo'),
    ('Federal Offices', 'DDPS', 'Office fédéral de la protection de la population', 'Office fédéral de la protection civile', 'OFPP', '', 'Bundesamt für Bevölkerungsschutz', 'Bundesamt für Zivilschutz', 'BABS', '', 'Federal Office for Civil Protection', 'FOCP'),
    ('Federal Offices', 'DDPS', 'Office fédéral du sport', '', 'OFSPO', '', 'Bundesamt für Sport', '', 'BASPO', '', 'Federal Office of Sport', 'FOSPO'),
    ('Federal Offices', 'DDPS', 'Office fédéral de la cybersécurité', '', 'OFCS', '', 'Bundesamt für Cybersicherheit', '', 'BACS', '', 'National Cyber Security Centre', 'NCSC'),
    ('OLD Federal Offices', 'DDPS', 'Forces aériennes', '', '', '', 'Luftwaffe', '', '', '', '', ''),
    ('OLD Federal Offices', 'DDPS', 'Forces terrestres', '', '', '', 'Heer', '', '', '', '', ''),
    ('OLD Federal Offices', 'DDPS', 'Etat-major général', '', '', '', 'Generalstab', '', '', '', '', ''),
    ('Federal Offices', 'FDF', 'Secrétariat général', '', 'SG-DFF', '', 'Generalsekretariat', '', 'GS-EFD', '', 'General Secretariat', 'GS-FDF'),
    ('Federal Offices', 'FDF', "Secrétariat d'état aux questions financières internationales", '', 'SFI', '', 'Staatssekretariat für internationale Finanzfragen', '', 'SIF', '', 'State Secretariat for International Finance', 'SIF'),
    ('Federal Offices', 'FDF', 'Administration fédérale des finances', '', 'AFF', '', 'Eidgenössische.n.r Finanzverwaltung', '', 'EFV', '', 'Federal Finance Administration', 'FFA'),
    ('Federal Offices', 'FDF', 'Office fédéral du personnel', '', 'OFPER', '', 'Eidgenössische.s.n Personalamt', '', 'EPA', '', 'Federal Office of Personnel', 'FOPER'),
    ('Federal Offices', 'FDF', 'Administration fédérale des contributions', '', 'AFC', '', 'Eidgenössische.n.r Steuerverwaltung', '', 'ESTV', '', 'Federal Tax Administration', 'FTA'),
    ('Federal Offices', 'FDF', 'Office fédéral de la douane et de la sécurité des frontières', 'Office fédéral des douanes', 'OFDF', '', 'Bundesamt für Zoll und Grenzsicherheit', 'Eidgenössische.n.r Zollverwaltung', 'BAZG', '', 'Federal Office for Customs and Border Security', 'FOCBS'),
    ('Federal Offices', 'FDF', 'Office fédéral de l’informatique et de la télécommunication', 'Office fédéral de l’informatique', 'OFIT', '', 'Bundesamt für Informatik und Telekommunikation', 'Bundesamt für Informatik', 'BIT', '', 'Federal Office of Information Technology, Systems and Telecommunication', 'FOITT'),
    ('Federal Offices', 'FDF', 'Office fédéral des constructions et de la logistique', '', 'OFCL', '', 'Bundesamt für Bauten und Logistik', '', 'BBL', '', 'Federal Office for Buildings and Logistics', 'FOBL'),
    ('OLD Federal Offices', 'FDF', 'Caisse fédérale d’assurance', 'Office fédéral des assurances privées', '', '', 'Eidgenössische.n.r Versicherungskasse', 'Bundesamt für Privatversicherungen', '', '', '', ''),
    ('OLD Federal Offices', 'FDF', 'Unité de pilotage informatique de la Confédération', '', 'UPIC', '', 'Informatiksteuerungsorgan des Bundes', '', 'ISB', '', '', ''),
    ('Federal Offices', 'EAER', 'Secrétariat général', '', 'SG-DEFR', '', 'Generalsekretariat', '', 'GS-WBF', '', 'General Secretariat', 'GS-EAER'),
    ('Federal Offices', 'EAER', 'Surveillance des prix', '', 'SPR', '', 'Preisüberwachung', '', 'PUE', '', 'Office of the Price Supervisor', 'PRS'),
    ('Federal Offices', 'EAER', 'Secrétariat d’Etat à l’économie', '', 'SECO', '', 'Staatssekretariat für Wirtschaft', '', 'SECO', '', 'State Secretariat for Economic Affairs', 'SECO'),
    ('Federal Offices', 'EAER', "Secrétariat d'Etat à la formation, à la recherche et à l’innovation", 'Office fédéral de la formation professionnelle et de la technologie', 'SEFRI', '', 'Staatssekretariat für Bildung, Forschung und Innovation', 'Bundesamt für Berufsbildung und Technologie', 'SBFI', '', 'State Secretariat for Education, Research and Innovation', 'SERI'),
    ('Federal Offices', 'EAER', 'Office fédéral de l’agriculture', '', 'OFAG', '', 'Bundesamt für Landwirtschaft', '', 'BLW', '', 'Federal Office for Agriculture', 'FOAG'),
    ('Federal Offices', 'EAER', 'Office fédéral pour l’approvisionnement économique du pays', '', 'OFAE', '', 'Bundesamt für wirtschaftliche Landesversorgung', '', 'BWL', '', 'Federal Office for National Economic Supply', 'FONES'),
    ('Federal Offices', 'EAER', 'Office fédéral du logement', '', 'OFL', '', 'Bundesamt für Wohnungswesen', '', 'BWO', '', 'Federal Office for Housing', 'FOH'),
    ('Federal Offices', 'EAER', 'Office fédéral du service civil', '', 'CIVI', '', 'Bundesamt für Zivildienst', '', 'ZIVI', '', 'Federal Office for Civilian Service', 'CIVI'),
    ('OLD Federal Offices', 'EAER', 'Office vétérinaire fédéral', '', '', '', 'Bundesamt für Veterinärwesen', '', '', '', '', ''),
    ('Federal Offices', 'DETEC', 'Secrétariat général', '', 'SG-DETEC', '', 'Generalsekretariat', '', 'GS-UVEK', '', 'General Secretariat', 'GS-DETEC'),
    ('Federal Offices', 'DETEC', 'Office fédéral des transports', '', 'OFT', '', 'Bundesamt für Verkehr', '', 'BAV', '', 'Federal Office of Transport', 'FOT'),
    ('Federal Offices', 'DETEC', 'Office fédéral de l’aviation civile', '', 'OFAC', '', 'Bundesamt für Zivilluftfahrt', '', 'BAZL', '', 'Federal Office of Civil Aviation', 'FOCA'),
    ('Federal Offices', 'DETEC', 'Office fédéral de l’énergie', '', 'OFEN', '', 'Bundesamt für Energie', '', 'BFE', '', 'Swiss Federal Office of Energy', 'SFOE'),
    ('Federal Offices', 'DETEC', 'Office fédéral des routes', '', 'OFROU', '', 'Bundesamt für Strassen', '', 'ASTRA', '', 'Federal Roads Office', 'FEDRO'),
    ('Federal Offices', 'DETEC', 'Office fédéral de la communication', '', 'OFCOM', '', 'Bundesamt für Kommunikation', '', 'BAKOM', '', 'Federal Office of Communications', 'OFCOM'),
    ('Federal Offices', 'DETEC', 'Office fédéral de l’environnement', 'Office fédéral de l’environnement, des forêts et du paysage', 'OFEV', '', 'Bundesamt für Umwelt', 'Bundesamt für Umwelt, Wald und Landschaft', 'BAFU', '', 'Federal Office for the Environment', 'FOEN'),
    ('Federal Offices', 'DETEC', 'Office fédéral du développement territorial', '', 'ARE', '', 'Bundesamt für Raumentwicklung', '', 'ARE', '', 'Federal Office for Spatial Development', 'ARE'),
    ('OLD Federal Offices', 'DETEC', 'Office fédéral des eaux et de la géologie', '', '', '', 'Bundesamt für Wasser und Geologie', '', '', '', '', ''),
]
