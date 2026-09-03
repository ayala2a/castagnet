"""Génère le rapport PDF du projet (HTML soigné -> Chrome headless -> PDF)."""

import base64
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ROOT = os.path.dirname(HERE)


def img(path, w="100%"):
    p = path if os.path.isabs(path) else os.path.join(FIG, path)
    if not os.path.exists(p):
        return ""
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img style="width:{w};max-width:100%;margin:8px 0;border:1px solid #ddd;border-radius:4px" src="data:image/png;base64,{b64}">'


CSS = """
@page { size: A4; margin: 20mm 16mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       font-size: 11pt; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 26pt; color: #14607a; margin: 0 0 4px; }
h2 { font-size: 16pt; color: #14607a; border-bottom: 2px solid #14607a;
     padding-bottom: 4px; margin-top: 26px; page-break-after: avoid; }
h3 { font-size: 12.5pt; color: #1f7a94; margin-top: 18px; page-break-after: avoid; }
p { margin: 7px 0; text-align: justify; }
.sub { font-size: 14pt; color: #1f7a94; margin-top: 6px; }
.meta { color: #666; font-size: 10pt; margin-bottom: 4px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }
th, td { border: 1px solid #cbd5db; padding: 5px 8px; text-align: left; }
th { background: #eaf3f6; }
code { background: #f2f4f5; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
.box { background: #f4f9fb; border-left: 4px solid #14607a; padding: 8px 14px;
       margin: 12px 0; border-radius: 0 4px 4px 0; }
.def { background: #fbfaf3; border-left: 4px solid #c9a227; padding: 6px 14px;
       margin: 10px 0; border-radius: 0 4px 4px 0; font-size: 10pt; }
.ok { color: #17803d; font-weight: 600; }
figure { margin: 10px 0; page-break-inside: avoid; }
figcaption { font-size: 9pt; color: #666; font-style: italic; text-align: center; }
.cover { text-align: center; padding: 60px 0 30px; }
.pill { display:inline-block; background:#14607a; color:#fff; border-radius:12px;
        padding:2px 12px; font-size:9pt; margin:2px; }
ul { margin: 6px 0; }
</style>
"""


BODY = f"""
<div class="cover">
  <h1>CastagNet</h1>
  <div class="sub">Un logiciel qui reconnaît automatiquement la qualité des châtaignes à partir de photos</div>
  <p class="meta" style="margin-top:24px">Mise en Situation Professionnelle (MESPR) — MSc BIHAR / ESTIA</p>
  <p class="meta">Auteur : <b>Mario Caballero</b> · Dépôt du code : github.com/ayala2a/castagnet</p>
  <div style="margin-top:18px">
    <span class="pill">Données</span>
    <span class="pill">Modèle</span>
    <span class="pill">Mise en production</span>
  </div>
</div>

<div class="box">
<b>Comment j'ai travaillé.</b> Je me suis aidé d'une intelligence artificielle (Claude) comme
assistant : pour analyser les données, retrouver les bonnes méthodes, écrire le code et corriger
les erreurs. Les choix, le raisonnement et la direction du projet restent les miens ; l'assistant
m'a surtout fait gagner du temps.
</div>

<h2>1. Le problème à résoudre</h2>
<p>Une coopérative corse (le GRPTMC) trie ses châtaignes sèches sur une machine automatique. La
machine prend chaque fruit en photo pendant qu'il défile, puis doit décider s'il est <b>bon
(Conforme)</b>, <b>mauvais (NON Conforme)</b>, d'une variété particulière à écarter
(<b>PIETRA</b>), ou si l'emplacement est simplement <b>vide</b>. On m'a confié le projet pour
l'améliorer sur trois points : la qualité des données, la fiabilité du logiciel de reconnaissance,
et sa capacité à tourner sur la vraie machine.</p>

<p>La règle imposée par la coopérative est simple à comprendre et guide tout mon travail :
<b>il vaut mieux jeter un bon fruit par erreur que d'en laisser passer un mauvais</b>. Laisser
passer un mauvais fruit gâche la farine et fait perdre le label qualité ; jeter un bon fruit fait
juste perdre un peu de marchandise. Concrètement, la coopérative demande que, parmi les fruits que
le logiciel déclare « bons », <b>au moins 95 % le soient vraiment</b> (peu d'erreurs), et que le
logiciel <b>retrouve au moins 85 % des bons fruits</b>. Le tout doit tourner en direct sur les
12 caméras de la machine, avec un ordinateur assez ancien.</p>

<div class="def"><b>Deux mots à retenir.</b> On parlera souvent de deux mesures :
la <b>précision</b> = « quand le logiciel dit bon, a-t-il raison ? » ; et le <b>rappel</b> =
« retrouve-t-il bien tous les bons fruits ? ». La coopérative veut une précision très haute
(≥ 95 %) et un bon rappel (≥ 85 %).</div>

<h2>2. Ma ligne de conduite</h2>
<p>Deux idées simples ont guidé tout le projet. La première : <b>une châtaigne, ce n'est pas une
photo mais deux</b> — la machine la photographie du dessus ET du dessous. La seconde : <b>dans le
doute, on écarte</b>. À partir de là, j'ai avancé dans l'ordre logique : d'abord bien comprendre et
nettoyer les données, ensuite construire un logiciel qui décide à partir des deux photos, enfin
vérifier qu'il tourne sur la vraie machine.</p>

<h2>3. Le travail sur les données</h2>

<h3>3.1 Récupérer les données et les vérifier</h3>
<p>J'ai d'abord récupéré les 35 254 photos fournies. En regardant de près, j'ai trouvé un problème
important : le rapport qui accompagnait les données ne parlait que de trois catégories et
oubliait complètement les emplacements <b>vides</b>. Or, en comparant l'étiquette d'origine
(devinée à partir du nom du fichier) avec la vraie étiquette vérifiée par des humains, j'ai
constaté que <b>36 % des photos étaient en réalité des emplacements vides</b>. Le rapport de
départ était donc faux ; je l'ai refait correctement.</p>
{img("01_avant_apres.png")}
<figure><figcaption>À gauche, l'ancien rapport (trois catégories) ; à droite, la réalité : les
emplacements « vides » représentent plus d'un tiers des photos.</figcaption></figure>
<p>J'ai aussi terminé la vérification d'un lot de photos qui restait à contrôler, et j'ai remarqué
que ce lot était très mal étiqueté au départ. Ça montre que les données n'étaient pas fiables telles
quelles — une chose que le sujet demandait justement d'analyser.</p>

<h3>3.2 Relier la photo du dessus et celle du dessous</h3>
<p>Puisqu'un fruit a deux photos, il faut savoir lesquelles vont ensemble. Je les ai reliées grâce
à leur nom de fichier. En vérifiant, j'ai découvert un point clé : quand les deux photos d'un même
fruit ne sont pas d'accord sur la catégorie, <b>c'est toujours parce que l'une des deux ne voit pas
le fruit</b> (il est hors du champ de cette caméra), jamais parce qu'elles voient deux qualités
différentes.</p>
{img("03_desaccords_TB.png")}
<figure><figcaption>Les seuls désaccords entre les deux photos viennent d'une vue « vide » ;
jamais d'un vrai conflit de qualité.</figcaption></figure>
<p><b>Pourquoi c'est important.</b> Ça prouve qu'on ne peut pas se fier à une seule photo :
environ un fruit sur trois semble « vide » sur l'une de ses deux photos alors qu'il est bien là.
La bonne règle est donc simple : <b>si une photo voit le fruit, on lui fait confiance</b>, et en cas
de vrai doute on choisit la catégorie la plus sévère — ce qui respecte la consigne « dans le doute,
on écarte ».</p>

<h3>3.3 Extraire et relier les images des vidéos</h3>
<p>On m'a aussi donné deux vidéos de 30 secondes qui filment le même fruit, une du dessus et une du
dessous, mais elles ne sont <b>pas parfaitement synchronisées</b> : l'une démarre un tout petit peu
avant l'autre. Il fallait retrouver ce décalage pour relier les bonnes images entre elles.</p>
<p><b>Pourquoi ne pas le faire à l'œil</b> : comparer deux vidéos de 750 images au jugé n'est ni
fiable ni sérieux. Je l'ai donc mesuré. J'ai repéré, dans chaque vidéo, les moments où un fruit
passe, et j'ai vu qu'ils passaient exactement au même rythme dans les deux — c'est donc bien le même
flux. En comparant ces moments, j'obtiens un décalage <b>clair et unique de 5 images (0,2 seconde)</b>.
J'ai vérifié une paire à l'écran (même fruit, même repère vert, vu sous deux angles), puis sorti une
vingtaine de paires propres.</p>
{img(os.path.join(ROOT, "data", "video_frames", "pairs_contact.png"))}
<figure><figcaption>Fruits reliés depuis les vidéos : en haut la première vidéo, en bas la seconde ;
chaque colonne est le même fruit vu des deux côtés.</figcaption></figure>

<h3>3.4 Préparer les données pour l'apprentissage</h3>
<p>Pour entraîner le logiciel, je raisonne par <b>fruit</b> (ses deux photos), pas par photo isolée.
Je répartis les fruits en trois groupes : un pour apprendre, un pour régler, un pour tester. Point
important : <b>les deux photos d'un même fruit restent toujours dans le même groupe</b>. Sinon, le
logiciel pourrait « reconnaître » un fruit déjà vu et tricher sans le vouloir — il aurait de bonnes
notes au test mais échouerait sur la vraie machine. J'ai vérifié que ce n'arrive jamais.</p>

<h2>4. Le logiciel de reconnaissance</h2>

<h3>4.1 Les différentes versions, et pourquoi</h3>
<p>J'ai construit plusieurs versions, de plus en plus fines, en me basant à chaque fois sur ce que
les résultats me montraient :</p>
<ul>
<li><b>Une version simple</b> qui ne regarde qu'une seule photo (la version de base demandée). Elle
rate presque tous les fruits PIETRA — logique, le défaut ne se voit souvent que d'un côté.</li>
<li><b>Une version qui regarde les deux photos ensemble.</b> C'est celle qui colle au problème :
deux photos, une décision. J'ai choisi un modèle de reconnaissance d'images <b>léger et rapide</b>
(appelé MobileNet), parce que la vraie machine a un ordinateur modeste. Je n'ai pas utilisé de
détection d'objet (type « encadrer le fruit ») : le fruit est déjà bien au centre, il suffit de le
classer.</li>
<li><b>Une version qui regarde en plus la différence entre les deux photos.</b> En voyant que les
résultats plafonnaient, je me suis dit : si un défaut n'est visible que d'un côté, alors les deux
photos se ressemblent moins à cet endroit. J'ai donc donné au logiciel non seulement les deux photos,
mais aussi <b>leur différence</b>. C'est ce qui a fait progresser le rappel.</li>
<li><b>Une version « radiale ».</b> Le sujet suggérait d'exploiter le fait que l'information est dans
le disque central de l'image. J'ai donc <b>déroulé ce disque à plat</b> (comme si on l'ouvrait en
partant du centre) avant de le montrer au logiciel. Avantage : peu importe comment le fruit est
tourné, ça revient au même. C'est finalement ma <b>meilleure version</b>.</li>
</ul>

<div class="def"><b>Une astuce utilisée partout.</b> Pour fiabiliser les décisions, je fais analyser
au logiciel plusieurs versions <b>pivotées</b> de la même photo, puis je prends la moyenne. Ça évite
qu'un mauvais angle fausse le résultat.</div>

<h3>4.2 Comment le logiciel décide (et pourquoi comme ça)</h3>
<p>Le logiciel ne se contente pas de choisir la catégorie la plus probable : ce serait risqué vu la
consigne. À la place, il ne déclare un fruit <b>« bon »</b> que s'il en est <b>vraiment sûr</b>
(au-dessus d'un niveau de confiance réglé exprès). C'est ce réglage qui garantit qu'au moins 95 %
des fruits déclarés bons le sont réellement, tout en en retrouvant le plus possible. J'ai d'ailleurs
choisi la meilleure version en visant directement cet objectif, pas une note globale.</p>

<h3>4.3 Les résultats (sur des fruits jamais vus)</h3>
<table>
<tr><th>Version</th><th>Bonnes réponses (global)</th><th>Précision / rappel sur « bon »</th><th>Objectif atteint ?</th></tr>
<tr><td>Version simple (1 photo)</td><td>66 %</td><td>95 % / 30 %</td><td>non</td></tr>
<tr><td>Deux photos ensemble</td><td>86 %</td><td>96 % / 77 %</td><td>non (rappel)</td></tr>
<tr><td>+ la différence entre les photos</td><td>91 %</td><td>95 % / 88 %</td><td class="ok">oui</td></tr>
<tr><td><b>Version radiale (retenue)</b></td><td><b>92 %</b></td><td><b>95 % / 92 %</b></td><td class="ok">oui, avec de la marge</td></tr>
</table>
<p>Le récit est clair : la version à une seule photo échoue (elle ne voit pas les défauts d'un
côté) ; regarder les deux photos aide beaucoup ; ajouter leur différence fait passer l'objectif ;
et la version radiale gagne encore. Chaque idée a été validée par les chiffres.</p>
{img("mlflow_comparatif.png")}
<figure><figcaption>Comparaison des différentes versions testées (suivi automatique des essais).</figcaption></figure>

<h3>4.4 Analyse des erreurs</h3>
{img("confusion_final.png", "60%")}
<figure><figcaption>Tableau des erreurs de la version retenue : les cases de la diagonale sont les
bonnes réponses, les autres les confusions.</figcaption></figure>
<p>Les erreurs qui restent sont surtout des confusions entre « bon » et « PIETRA » (des défauts de
surface très fins). Le plus rassurant pour la coopérative : <b>les erreurs vont dans le bon sens</b>
— le logiciel écarte parfois un bon fruit par excès de prudence, mais laisse très rarement passer un
mauvais. C'est exactement ce que demande la consigne.</p>

<h3>4.5 Vérifier que le résultat est solide</h3>
<p>Pour être sûr que ces résultats ne sont pas un coup de chance, j'ai <b>refait l'expérience 5 fois</b>
sur 5 découpages différents des données. Les résultats restent stables (précision autour de 95 %,
rappel autour de 87 %, avec très peu de variation). Le logiciel est donc fiable, pas chanceux.</p>
{img("mlflow_history.png")}
<figure><figcaption>Évolution des performances au fil de l'apprentissage : ça tâtonne au début,
puis ça se stabilise au-dessus des objectifs.</figcaption></figure>

<h2>5. Faire tourner le logiciel sur la vraie machine</h2>
<p>J'ai converti le logiciel dans un format standard et léger (appelé ONNX), et vérifié qu'il donne
exactement les mêmes réponses qu'avant conversion. Son coût est très faible :</p>
<table>
<tr><th>Élément</th><th>Valeur</th></tr>
<tr><td>Taille du logiciel</td><td>~15 Mo</td></tr>
<tr><td>Mémoire utilisée (au repos / en marche)</td><td>~75 Mo / ~220 Mo</td></tr>
<tr><td>Temps de réponse pour un cycle de la machine</td><td>~0,05 seconde</td></tr>
</table>
<p>La carte graphique de la machine dispose de 3 Go de mémoire : on en utilise à peine un dixième,
il y a donc une marge très confortable. Le temps de réponse est largement suffisant pour suivre la
cadence de tri. En résumé, <b>toutes les exigences sont respectées</b>, et la mémoire est le point le
plus tranquille. À l'usage, il suffit de donner au logiciel les deux photos d'un fruit pour obtenir
sa catégorie.</p>

<h2>6. Ce que j'ai utilisé</h2>
<p><b>Outils.</b> Le langage Python ; des bibliothèques spécialisées pour créer et entraîner le
logiciel de reconnaissance (PyTorch, MobileNet) ; un outil de traitement d'images (OpenCV) pour
découper les vidéos et préparer les photos ; un outil pour découper proprement les données
(scikit-learn) ; un carnet de bord automatique des essais (MLflow) ; un format standard pour la mise
en production (ONNX) ; l'outil de stockage des images fourni (DVC) et l'outil de vérification des
étiquettes (Streamlit) ; et Git pour archiver tout le code.</p>
<p><b>Ce qui m'a inspiré.</b> Je me suis appuyé sur des projets publics reconnus de tri de
fruits et de graines par caméra, qui utilisent la même idée de « plusieurs photos par objet », pour
confirmer ma direction.</p>

<h2>7. Conclusion</h2>
<p>Le logiciel final <b>respecte la consigne avec de la marge</b> (95 % de précision et 92 % de
rappel sur les fruits jamais vus), tient largement sur la machine de la coopérative, et son résultat
est prouvé stable. L'essentiel du travail n'a pas été d'empiler des calculs, mais d'avoir <b>bien
compris le problème</b> : une châtaigne se juge sur ses deux faces, un défaut d'un seul côté se
repère par la différence entre les photos, et dérouler le disque de l'image aide à mieux voir. En
gardant toujours la consigne de la coopérative en tête — la sûreté avant tout — j'ai évité de courir
après un beau chiffre qui n'aurait pas tenu en vrai.</p>
<p><b>Pistes pour la suite.</b> Vérifier si une partie des confusions « bon / PIETRA » vient d'erreurs
d'étiquetage dans les données ; mesurer le temps de réponse réel sur la machine de la coopérative ; et,
si besoin d'un tout petit gain, combiner plusieurs versions du logiciel.</p>
"""

HTML = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}<body>{BODY}</body></html>"

out_html = os.path.join(HERE, "_rapport_tmp.html")
open(out_html, "w").write(HTML)

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out_pdf = os.path.join(HERE, "RAPPORT_CastagNet.pdf")
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}", "file://" + out_html], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(out_html)
print("PDF écrit:", out_pdf, os.path.getsize(out_pdf) // 1024, "Ko")
