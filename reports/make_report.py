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
  <div class="sub">Un logiciel qui reconnaît la qualité des châtaignes à partir de photos</div>
  <p class="meta" style="margin-top:24px">Mise en Situation Professionnelle — MSc BIHAR / ESTIA</p>
  <p class="meta">Auteur : <b>Mario Caballero</b> · Code : github.com/ayala2a/castagnet</p>
</div>

<div class="box">
<b>Comment j'ai travaillé.</b> Je me suis aidé d'une intelligence artificielle (Claude) comme
assistant, pour analyser les données, écrire le code et corriger les erreurs. Les choix et la
direction du projet restent les miens.
</div>

<h2>1. Le problème</h2>
<p>Une coopérative corse trie ses châtaignes sèches sur une machine automatique qui photographie
chaque fruit du dessus et du dessous. Il faut décider s'il est <b>bon</b>, <b>mauvais</b>, d'une
variété à écarter (<b>PIETRA</b>), ou si l'emplacement est <b>vide</b>. La règle de la coopérative
guide tout : <b>mieux vaut jeter un bon fruit que laisser passer un mauvais</b> (un mauvais fruit
gâche la farine). Concrètement : parmi les fruits déclarés « bons », au moins 95 % doivent l'être
vraiment, et il faut retrouver au moins 85 % des bons — en direct, sur un ordinateur modeste.</p>
<div class="def"><b>Deux mots à retenir.</b> <b>Précision</b> = « quand le logiciel dit bon, a-t-il
raison ? ». <b>Rappel</b> = « retrouve-t-il bien tous les bons fruits ? ».</div>

<h2>2. Le travail sur les données</h2>
<p>En vérifiant les 35 254 photos, j'ai trouvé que le rapport fourni était faux : il oubliait les
emplacements <b>vides</b>, qui représentent en réalité <b>36 % des photos</b>. Je l'ai refait
correctement, et terminé la vérification d'un lot mal étiqueté.</p>
{img("01_avant_apres.png")}
<figure><figcaption>À gauche l'ancien rapport (incomplet), à droite la réalité avec les emplacements vides.</figcaption></figure>
<p><b>Une châtaigne = deux photos.</b> J'ai relié la vue du dessus et celle du dessous de chaque
fruit. Découverte importante : quand les deux photos ne sont pas d'accord, c'est toujours parce que
l'une ne voit pas le fruit — jamais un vrai conflit de qualité. Donc <b>on ne peut pas se fier à une
seule photo</b> (un fruit sur trois paraît « vide » sur une de ses vues), il faut regarder les deux.</p>
<p><b>Les vidéos.</b> Deux vidéos filment le même fruit du dessus et du dessous, mais légèrement
décalées dans le temps. Plutôt que d'aligner « à l'œil », j'ai mesuré le décalage en repérant les
moments où les fruits passent : c'est un décalage <b>net de 5 images (0,2 s)</b>. J'ai ainsi relié
les fruits d'une vidéo à l'autre.</p>
{img(os.path.join(ROOT, "data", "video_frames", "pairs_contact.png"))}
<figure><figcaption>Fruits reliés depuis les vidéos : chaque colonne est le même fruit, vu du dessus (haut) et du dessous (bas).</figcaption></figure>
<p>Enfin, pour l'entraînement, je répartis les fruits en trois groupes (apprendre / régler / tester)
en gardant <b>toujours les deux photos d'un fruit dans le même groupe</b> — sinon le logiciel pourrait
« reconnaître » un fruit déjà vu et tricher sans le vouloir.</p>

<h2>3. Le logiciel et ses résultats</h2>
<p>J'ai construit plusieurs versions, de plus en plus fines, guidé par les résultats :</p>
<ul>
<li><b>Une version simple</b> (une seule photo) : elle rate presque tous les PIETRA, car le défaut
ne se voit souvent que d'un côté.</li>
<li><b>Deux photos ensemble</b>, avec un modèle léger et rapide (adapté à la machine modeste).</li>
<li><b>+ la différence entre les deux photos</b> : si un défaut n'est visible que d'un côté, les deux
photos se ressemblent moins à cet endroit. Cette idée a débloqué le rappel.</li>
<li><b>Version « radiale »</b> : je déroule le disque de l'image à plat avant de l'analyser. Peu
importe comment le fruit est tourné, ça revient au même. C'est ma <b>meilleure version</b>.</li>
</ul>
<p>Pour décider, le logiciel ne déclare « bon » que s'il en est <b>vraiment sûr</b> — c'est ce
réglage qui garantit la précision de 95 % demandée.</p>
<table>
<tr><th>Version</th><th>Bonnes réponses</th><th>Précision / rappel sur « bon »</th><th>Objectif</th></tr>
<tr><td>Version simple (1 photo)</td><td>66 %</td><td>95 % / 30 %</td><td>non</td></tr>
<tr><td>Deux photos ensemble</td><td>86 %</td><td>96 % / 77 %</td><td>non</td></tr>
<tr><td>+ différence entre les photos</td><td>91 %</td><td>95 % / 88 %</td><td class="ok">oui</td></tr>
<tr><td><b>Version radiale (retenue)</b></td><td><b>92 %</b></td><td><b>95 % / 92 %</b></td><td class="ok">oui, avec marge</td></tr>
</table>
<p>Les erreurs qui restent sont des confusions « bon / PIETRA » (défauts très fins), et surtout
elles <b>vont dans le bon sens</b> : le logiciel écarte parfois un bon fruit par prudence, mais
laisse très rarement passer un mauvais. Pour être sûr que ce n'est pas un coup de chance, j'ai
<b>refait l'expérience 5 fois</b> : les résultats restent stables.</p>

<h2>4. Sur la vraie machine</h2>
<p>J'ai converti le logiciel dans un format standard et léger, et vérifié qu'il donne les mêmes
réponses. Il pèse ~15 Mo, utilise peu de mémoire (~220 Mo en marche, pour 3 Go disponibles) et
répond en ~0,05 seconde par cycle. <b>Toutes les exigences sont respectées</b>, avec une marge
confortable. À l'usage, on lui donne les deux photos d'un fruit et il renvoie sa catégorie.</p>

<h2>5. Conclusion</h2>
<p>Le logiciel final <b>respecte la consigne avec de la marge</b> (95 % de précision, 92 % de rappel
sur des fruits jamais vus), tient largement sur la machine, et son résultat est prouvé stable.
L'essentiel n'a pas été d'empiler des calculs, mais d'avoir <b>bien compris le problème</b> : une
châtaigne se juge sur ses deux faces. En gardant toujours la règle « la sûreté avant tout », j'ai
évité de courir après un chiffre qui n'aurait pas tenu en vrai.</p>
<p style="color:#666;font-size:9.5pt;margin-top:16px"><i>Outils utilisés : Python, PyTorch,
OpenCV, MLflow, ONNX, DVC, Git. Détails techniques dans l'annexe du dépôt.</i></p>
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
