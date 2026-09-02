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
.lead { font-size: 12pt; color: #444; }
.meta { color: #666; font-size: 10pt; margin-bottom: 4px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }
th, td { border: 1px solid #cbd5db; padding: 5px 8px; text-align: left; }
th { background: #eaf3f6; }
code { background: #f2f4f5; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f7f8f9; border: 1px solid #e2e6e8; border-radius: 5px;
      padding: 8px 10px; font-size: 9pt; overflow-x: auto; }
.box { background: #f4f9fb; border-left: 4px solid #14607a; padding: 8px 14px;
       margin: 12px 0; border-radius: 0 4px 4px 0; }
.ok { color: #17803d; font-weight: 600; }
figure { margin: 10px 0; page-break-inside: avoid; }
figcaption { font-size: 9pt; color: #666; font-style: italic; text-align: center; }
.cover { text-align: center; padding: 60px 0 30px; }
.cover .sub { font-size: 14pt; color: #1f7a94; margin-top: 6px; }
.pill { display:inline-block; background:#14607a; color:#fff; border-radius:12px;
        padding:2px 12px; font-size:9pt; margin:2px; }
ul { margin: 6px 0; }
"""


BODY = f"""
<div class="cover">
  <h1>CastagNet</h1>
  <div class="sub">Classification de châtaignes sèches par vision — rapport de projet</div>
  <p class="meta" style="margin-top:24px">Mise en Situation Professionnelle (MESPR) — MSc BIHAR / ESTIA</p>
  <p class="meta">Auteur : ayala2a · Dépôt : github.com/ayala2a/castagnet</p>
  <div style="margin-top:18px">
    <span class="pill">C28 · problématique data</span>
    <span class="pill">C29 · modèles</span>
    <span class="pill">C30 · production</span>
    <span class="pill">C31 · organisation données</span>
    <span class="pill">C32 · programme IA</span>
  </div>
</div>

<div class="box">
<b>Note de méthode.</b> Pour ce projet, je me suis appuyé sur une IA générative (Claude)
comme copilote : pour auditer le dataset, retrouver les bonnes références techniques,
écrire le code et débugger. Les décisions, le raisonnement et les choix de direction
présentés ici sont les miens ; l'IA a surtout accéléré la recherche et la mise en œuvre.
</div>

<h2>1. Contexte et problématique</h2>
<p>La coopérative GRPTMC exploite une ligne de tri automatique de châtaignes sèches
destinées, notamment, à la Farine de Châtaigne Corse — une filière sous signe de qualité
où la rigueur du tri conditionne la conformité du produit fini. La machine analyse le flux
en continu via <b>12 caméras</b> (6 postes, une vue du dessus et une du dessous par poste)
et doit classer chaque fruit en quatre catégories : <b>Conforme, NON Conforme, PIETRA,
Vide</b>. On m'a confié le projet en l'état, avec pour mission de le faire progresser sur
trois fronts : la qualité de la donnée, la robustesse du modèle, et la compatibilité avec
les contraintes de production.</p>

<p>Le cahier des charges fixe des exigences <b>asymétriques</b>, et c'est le fil directeur
de tout mon travail : laisser passer un mauvais fruit dans le lot Conforme dégrade la
farine et menace la certification, alors qu'écarter à tort un bon fruit ne coûte qu'un peu
de rendement. La coopérative demande donc, sur la classe Conforme, une <b>précision ≥ 95 %
</b> (peu de mauvais fruits acceptés) et un <b>rappel ≥ 85 %</b>, le tout en <b>temps réel
sur les 12 flux</b> et sur un matériel modeste (une ancienne carte GeForce GTX 1060 de
3 Go). Chaque décision technique de ce rapport découle de ces contraintes.</p>

<h2>2. Le fil directeur</h2>
<p>Deux idées structurent tout le projet. La première vient du terrain : <b>une châtaigne
n'est pas une image, c'est une paire d'images</b> (une vue du dessus, une du dessous). La
seconde vient du cahier des charges : <b>on préfère toujours écarter un bon fruit que
laisser passer un mauvais</b>. À partir de là, j'ai déroulé le projet dans l'ordre logique
— d'abord comprendre et fiabiliser la donnée, ensuite construire un modèle qui décide sur
la paire et respecte l'asymétrie précision/rappel, enfin vérifier qu'il tient en
production.</p>

<h2>3. Travail sur la donnée (§4.1)</h2>

<h3>3.1 Récupération et audit</h3>
<p>Les images (35 254 au total) sont versionnées par DVC et stockées sur un drive partagé
— j'ai donc commencé par les rapatrier. En creusant les labels, j'ai découvert un problème
important : le rapport statistique fourni ne connaissait que trois classes et ignorait
complètement la classe <b>« Vide »</b>. En croisant le label déduit du nom de fichier avec
la vraie cible relue, j'ai constaté que <b>35,6 % des images avaient été reclassées en
Vide</b> lors de la relecture (un créneau photographié peut être vide). Le rapport de
départ était donc périmé ; je l'ai refait sur les vrais labels.</p>
{img("01_avant_apres.png")}
<figure><figcaption>Répartition des labels : le rapport fourni (à gauche) ignore la classe
Vide, qui représente pourtant 36 % des images une fois la relecture prise en compte.</figcaption></figure>
<p>J'ai aussi terminé la relecture du reliquat 2026 et remarqué au passage que ce lot était
mal étiqueté à l'acquisition (à peine 5 % de vrais PIETRA dedans) — un bon exemple de
l'hétérogénéité du diagnostic, que le sujet demandait justement d'analyser.</p>

<h3>3.2 Relier les deux vues (dataset)</h3>
<p>Puisqu'un fruit = deux photos, j'ai relié les vues T et B via une clé simple : le nom de
fichier privé de la position caméra. Cela reconstitue <b>19 078 châtaignes</b>. J'ai ensuite
analysé les cas où les deux vues étaient en désaccord, et le résultat est net :
<b>100 % des désaccords viennent d'une face vue comme « Vide »</b> — jamais un vrai conflit
entre deux classes.</p>
{img("03_desaccords_TB.png")}
<figure><figcaption>Composition des couples de vues T/B : les seuls écarts sont « une face
Vide », jamais deux vraies classes différentes.</figcaption></figure>
<p><b>Pourquoi c'est décisif.</b> Cela prouve que décider sur une seule image est trompeur :
environ un tiers des fruits présents apparaissent « Vide » sur l'une de leurs deux vues. J'en
tire une règle d'agrégation simple et sûre — <b>la classe non-Vide gagne</b>, et en cas de
vrai doute je prends la plus sévère — qui colle à la priorité « pureté du lot » du cahier des
charges.</p>

<h3>3.3 Extraction et appariement des vidéos</h3>
<p>Deux vidéos de 30 s filment le même créneau, une vue du haut et une du bas, mais avec un
micro-décalage temporel. Il fallait les apparier. <b>Pourquoi ne pas le faire à l'œil</b> :
aligner deux flux de 750 images au jugé n'est ni fiable ni reproductible. J'ai donc mesuré
le décalage. J'ai détecté dans chaque vidéo la <b>séquence d'arrivée des châtaignes</b>, et
constaté qu'elles avaient exactement la même cadence (une châtaigne toutes les 27 images) —
donc le même flux. <b>Pourquoi ce signal plutôt qu'un signal d'image global</b> : les deux
caméras cadrent différemment, et mes premiers essais de corrélation globale donnaient des
décalages incohérents (de −102 à +86 images) ; la suite des arrivées, elle, est une
information commune et robuste. En alignant les deux séquences, j'obtiens un décalage
<b>net et unique de 5 images (0,2 s)</b>, la vue du bas en avance. J'ai vérifié une paire à
l'écran — même fruit, même repère vert, deux angles — puis sorti une vingtaine de paires
propres en prenant, pour chaque fruit, l'image où il est le mieux visible dans chaque vidéo.</p>
{img(os.path.join(ROOT, "data", "video_frames", "pairs_contact.png"))}
<figure><figcaption>Paires T/B reconstituées depuis les vidéos (haut = vidéo A, bas = vidéo B,
chaque colonne = une même châtaigne sous deux angles).</figcaption></figure>

<h3>3.4 Dataset final et découpage anti-fuite</h3>
<p>Pour l'entraînement, l'unité est la <b>châtaigne</b> (sa paire), pas l'image. Le
découpage train/val/test est <b>stratifié</b> (par classe et par année) et surtout
<b>groupé par châtaigne</b> : les deux vues d'un même fruit tombent toujours dans le même
sous-ensemble. <b>Pourquoi</b> : sinon la vue T d'un fruit pourrait être en entraînement et
sa vue B en test, et le modèle « reconnaîtrait le fruit » (fuite de données) — des scores
flatteurs en test mais un échec en production. J'ai vérifié : zéro fuite. Le sous-ensemble
vidéo, non labellisé, n'est pas injecté dans l'entraînement (l'ajouter dégraderait le
signal) ; il constitue un livrable d'extraction/appariement à part.</p>

<h2>4. Modélisation (§4.2)</h2>

<h3>4.1 Les architectures, et pourquoi</h3>
<p>J'ai comparé plusieurs modèles, dans une progression logique guidée par les résultats :</p>
<ul>
<li><b>Un CNN « maison »</b> (codé à la main, sur une seule image) comme baseline exigée.
Il plafonne et rate presque totalement PIETRA — logique, un défaut ne se voit souvent que
d'un côté.</li>
<li><b>Un réseau dual-branch</b> : deux extracteurs de caractéristiques à poids partagés
(siamois) sur les vues T et B, puis une fusion. C'est l'architecture qui correspond au
problème (deux vues → une décision). <b>Pourquoi un backbone léger (MobileNetV3)</b> :
la cible est une GTX 1060 de 3 Go qui doit traiter les 12 flux — il faut rester rapide et
compact. <b>Pourquoi pas de la détection d'objets (YOLO)</b> : la châtaigne est déjà centrée
dans le créneau, on n'a qu'à la classer, pas à la localiser.</li>
<li><b>Une fusion enrichie <code>[T, B, |T−B|]</code></b>. En observant que tout plafonnait
vers 0,86–0,88 sur mon indicateur clé, j'ai raisonné sur ce que « voit » le réseau : si un
défaut n'apparaît que d'un côté, les caractéristiques des deux vues deviennent différentes à
cet endroit. Plutôt que de simplement concaténer, j'ai donné explicitement au modèle
<b>l'écart absolu entre les vues</b>. C'est ce qui a débloqué le rappel.</li>
<li><b>Une représentation radiale</b> (piste « pour aller plus loin » du sujet) : je déroule
le disque en coordonnées polaires avant le réseau. La cohérence radiale s'aligne alors sur un
axe, et une rotation du fruit devient un simple décalage — robustesse à l'orientation par
construction. C'est finalement mon <b>meilleur modèle</b>.</li>
</ul>

<h3>4.2 La métrique de décision, et pourquoi</h3>
<p>Je ne me fie pas à l'accuracy globale (trompeuse en déséquilibre) ni à l'argmax brut.
Comme le cahier des charges impose une contrainte asymétrique, je <b>calibre un seuil de
confiance</b> sur la classe Conforme : on n'accepte « Conforme » que si la probabilité
dépasse ce seuil. C'est ce mécanisme qui garantit la précision ≥ 95 % tout en maximisant le
rappel. J'ai d'ailleurs sélectionné le meilleur modèle non pas sur l'accuracy, mais
directement sur le <b>rappel Conforme à précision 95 %</b> — l'objectif métier réel.</p>

<h3>4.3 Résultats (jeu de test)</h3>
<table>
<tr><th>Modèle</th><th>Accuracy</th><th>Conforme précision / rappel</th><th>Cahier des charges</th></tr>
<tr><td>CNN maison (baseline, 1 vue)</td><td>0,659</td><td>0,954 / 0,296</td><td>non</td></tr>
<tr><td>Dual-branch (concaténation)</td><td>0,861</td><td>0,955 / 0,774</td><td>non (rappel)</td></tr>
<tr><td>Dual-branch + fusion |T−B| + TTA</td><td>0,913</td><td>0,953 / 0,884</td><td class="ok">oui</td></tr>
<tr><td><b>Dual-branch radial + TTA (retenu)</b></td><td><b>0,923</b></td><td><b>0,951 / 0,925</b></td><td class="ok">oui, avec marge</td></tr>
</table>
{img("mlflow_comparatif.png")}
<figure><figcaption>Comparatif des runs suivis dans MLflow.</figcaption></figure>
<p>La progression raconte l'histoire : la baseline mono-vue échoue (rappel PIETRA de 0,17,
car le défaut ne se voit que d'un côté) ; les deux vues font passer PIETRA à ~0,85 ; la
fusion <code>|T−B|</code> débloque le rappel Conforme au-dessus de 85 % ; et la
représentation radiale gagne encore ~4 points de rappel (0,884 → 0,925) à précision
équivalente, confirmant l'intuition du sujet sur la cohérence radiale.</p>

<h3>4.4 Analyse des erreurs et de l'entraînement</h3>
{img("confusion_final.png", "62%")}
<figure><figcaption>Matrice de confusion du modèle final sur le jeu de test.</figcaption></figure>
<p>Le point dur résiduel est la confusion Conforme ↔ PIETRA (défauts de surface fins). Fait
important pour la filière : <b>les erreurs vont dans le sens sûr</b> — le modèle recale à
tort quelques bons fruits (perte de rendement, tolérée) bien plus qu'il ne laisse passer de
mauvais. C'est exactement la philosophie du cahier des charges.</p>
{img("mlflow_history.png")}
<figure><figcaption>Historique d'entraînement : oscillation initiale puis stabilisation
(cosine LR) ; précision Conforme au-dessus de 0,95, rappel autour de la cible.</figcaption></figure>

<h3>4.5 Validation croisée (robustesse)</h3>
<p>Pour ne pas me fier à un seul découpage, j'ai mené une validation croisée en 5 plis
(toujours groupée par châtaigne). Résultats stables : précision Conforme <b>0,949 ± 0,012</b>,
rappel <b>0,868 ± 0,019</b>, accuracy <b>0,887 ± 0,012</b>. L'écart-type de 1 à 2 points
confirme que le résultat n'est pas un coup de chance. Ces modèles de validation ne sont pas
le modèle livré — ils ne servent qu'à en éprouver la fiabilité.</p>

<h2>5. Compatibilité production (§4.3)</h2>
<p>J'ai exporté le modèle retenu au format <b>ONNX</b> (avec un axe de lot dynamique pour
traiter d'un coup les images d'un cycle) et vérifié que ses sorties correspondent à celles de
PyTorch. Le coût est très faible :</p>
<table>
<tr><th>Poste</th><th>Valeur</th></tr>
<tr><td>Paramètres</td><td>3,71 M</td></tr>
<tr><td>Poids ONNX (FP32)</td><td>~15,3 Mo (≈ 7,7 Mo en FP16)</td></tr>
<tr><td>Mémoire au repos / par cycle (6 paires)</td><td>~75 Mo / ~220 Mo</td></tr>
<tr><td>Latence par cycle (CPU de dev)</td><td>~48 ms (~240 images/s)</td></tr>
</table>
<p>Sur les 3 Go de la GTX 1060, la marge est d'un facteur ~10 ; sur GPU en FP16, la latence
sera bien plus basse. <b>Recommandation</b> : retenir le dual-branch radial + TTA. Toutes les
exigences (précision, rappel, temps réel, tenue en 3 Go) sont satisfaites, la mémoire étant le
point le plus confortable. L'usage se fait via <code>predict.py</code>, qui prend les deux
photos d'un fruit et renvoie sa classe.</p>

<h2>6. Technologies et ressources utilisées</h2>
<p><b>Outils.</b> Python 3.13 ; PyTorch (accélération MPS sur Mac) et torchvision
(MobileNetV3) pour les modèles ; OpenCV pour l'extraction vidéo, le masque circulaire et le
déroulé polaire ; scikit-learn (<code>StratifiedGroupKFold</code>) pour le découpage
anti-fuite ; MLflow pour le suivi des expériences ; ONNX / onnxruntime pour l'export et la
mesure de latence ; DVC pour récupérer le dataset ; l'outil Streamlit fourni pour terminer la
labellisation ; Git pour le dépôt de travail.</p>
<p><b>Références qui ont orienté mes choix.</b> Des templates PyTorch de référence pour la
structure du projet ; des travaux de tri de fruits/graines multi-caméra (grading de pommes
multi-vues, SeedSortNet) qui m'ont conforté dans l'idée du dual-branch ; la bibliothèque timm
pour les backbones légers.</p>

<h2>7. Conclusion et perspectives</h2>
<p>Le projet aboutit à un modèle qui <b>respecte le cahier des charges avec marge</b> sur le
jeu de test (précision Conforme 95,1 %, rappel 92,5 %, accuracy 92,3 %), tient largement sur
le matériel cible, et dont le résultat est prouvé stable par validation croisée. L'apport
principal n'est pas d'avoir empilé des couches, mais d'avoir <b>traduit le problème métier en
modèle</b> : une châtaigne se juge sur ses deux faces, un défaut crée une asymétrie entre les
vues, et la représentation radiale exploite la géométrie circulaire du créneau. Piloter chaque
choix par la contrainte du cahier des charges — la précision d'abord, sur du matériel modeste
— m'a évité d'optimiser un chiffre qui n'aurait pas tenu en production.</p>
<p><b>Perspectives.</b> Vérifier la part de bruit d'étiquetage dans la confusion
Conforme/PIETRA (potentiellement plus rentable qu'un changement d'architecture) ; mesurer la
latence réelle sur la GTX 1060 cible ; et envisager un ensemble de modèles si un léger gain de
précision était requis, en pesant le surcoût de latence.</p>
"""

HTML = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{BODY}</body></html>"

out_html = os.path.join(HERE, "RAPPORT_CastagNet.html")
open(out_html, "w").write(HTML)
print("HTML écrit:", out_html)

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
out_pdf = os.path.join(HERE, "RAPPORT_CastagNet.pdf")
subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}", "file://" + out_html], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("PDF écrit:", out_pdf, os.path.getsize(out_pdf) // 1024, "Ko")
