# CastagNet — ma démarche

*Note de méthode : pour ce projet je me suis appuyé sur une IA générative (Claude)
comme copilote — pour auditer le dataset, retrouver les bons repos de référence,
écrire le pipeline et débugger. Je décris ci-dessous ce que j'ai fait, pourquoi, et
ce qui m'a mis sur chaque piste. Les décisions et les raisons sont les miennes ;
l'IA a surtout accéléré la recherche et la mise en œuvre.*

---

## Par quoi j'ai commencé

On m'a remis un dépôt `castagnia_data` sous forme de zip : des labels dans deux CSV,
un outil Streamlit de labellisation, un rapport statistique, deux vidéos, et surtout
un dossier `images/` qui n'était pas là — juste un fichier `images.dvc`. C'est ce
fichier, plus le README, qui m'a mis sur la piste de **DVC** (Data Version Control) :
les 35 254 images sont versionnées et stockées sur un Drive partagé, pas dans le zip.
J'ai donc installé DVC et fait un `dvc pull` pour rapatrier le dataset. Ça a coincé
plusieurs fois (DVC veut un dépôt git, un conflit de versions `pyOpenSSL`/`cryptography`,
une appli OAuth Google en mode test), mais une fois débloqué j'avais mes ~975 Mo
d'images en local, cohérentes à 100 % avec le CSV.

## Le piège du dataset

En regardant les données de près, je suis tombé sur un truc important : le rapport
fourni ne parlait que de 3 classes (Conforme / NON Conforme / PIETRA), alors que la
cible d'entraînement en a 4, avec **Vide**. Ce qui m'a mis la puce à l'oreille, c'est
l'écart entre deux colonnes : `label_filename` (le label déduit du nom de fichier) et
`label_principal` (la vraie cible relue). En croisant les deux, j'ai vu que **35,6 %
des images avaient été reclassées en « Vide »** à la relecture — un créneau
photographié peut être vide. Conclusion : le rapport fourni était périmé, je l'ai
refait sur les vrais labels. J'ai aussi terminé la relecture qui restait (le lot
PIETRA / caméra 6 / 2026) avec l'outil Streamlit — et au passage j'ai constaté que ce
lot était mal étiqueté à l'acquisition (à peine 5 % de vrais PIETRA dedans).

## L'idée qui structure tout : une châtaigne = deux photos

La machine filme chaque fruit par-dessus (T) et par-dessous (B). Donc décider sur une
seule image, c'est se priver de la moitié de l'info. Ce qui m'a vraiment convaincu,
c'est l'analyse des paires T/B : quand j'ai regardé les cas où les deux vues n'étaient
pas d'accord, **100 % de ces désaccords venaient d'une face vue comme « Vide »** —
jamais un vrai conflit entre deux classes. Autrement dit, quand le fruit est là, les
deux caméras sont d'accord ; la seule « divergence » c'est quand une caméra ne le
capte pas. Ça donne une règle d'agrégation simple : **la classe non-Vide gagne**, et
en cas de vrai doute on prend la plus sévère (on préfère rejeter une bonne châtaigne
que laisser passer une mauvaise — c'est exactement ce que demande le cahier des
charges pour protéger la pureté du lot).

Pour relier les deux vues, j'ai utilisé une clé toute bête : le nom de fichier privé
de la position `T/B`. Ça reconstitue 19 078 châtaignes (dont 16 176 paires complètes).

## Comment j'ai monté le pipeline

Côté outils, j'ai fait des choix guidés par la contrainte de production (le modèle doit
tourner sur une vieille GTX 1060 de 3 Go et traiter 12 flux en temps réel). Pour ne pas
réinventer la roue, j'ai regardé ce que font les projets sérieux du domaine. Ce qui m'a
orienté :

- **La structure de projet** — je me suis inspiré de `lightning-hydra-template` (le
  template PyTorch de référence) pour séparer proprement données / modèles /
  entraînement, même si j'ai gardé une boucle PyTorch « nue » plutôt que Lightning,
  pour que le code reste lisible et défendable.
- **L'architecture 2 vues** — des travaux sur le tri de fruits multi-caméra (un papier
  de grading de pommes multi-view, et SeedSortNet sur le tri de graines) m'ont conforté
  dans l'idée d'un réseau **dual-branch** : deux extracteurs partagés sur T et B, puis
  fusion.
- **Le backbone** — j'ai pris **MobileNetV3-Small puis Large** (via `torchvision`),
  parce que c'est léger et rapide : parfait pour la 1060 et pour empiler les 24 images
  d'un tick en un seul passage.
- **Le prétraitement** — l'info utile est dans le disque central du slot, donc je
  recadre au centre et j'applique un **masque circulaire** (avec OpenCV) pour virer le
  fond. Et comme le disque est invariant en rotation, j'augmente en tournant les images
  de 0 à 360° — une augmentation quasi gratuite.
- **Le découpage train/val/test** — le point piège quand un objet a plusieurs images,
  c'est la fuite de données : il ne faut jamais que la vue T d'un fruit soit en train
  et sa vue B en test, sinon le modèle « reconnaît le fruit » et les scores sont
  bidons. J'ai donc groupé par châtaigne avec `StratifiedGroupKFold` de scikit-learn,
  en stratifiant aussi par classe et par année. J'ai vérifié : zéro fuite.
- **Le suivi** — tous mes runs sont tracés dans **MLflow** (métriques par classe,
  matrices de confusion), ce qui m'a permis de comparer les modèles à égalité.

## L'itération, et ce qui a fini par débloquer

Ma baseline (un petit CNN codé à la main sur une seule image) plafonne vers 66 % et
rate complètement PIETRA — logique, un défaut ne se voit souvent que d'un côté. Le
dual-branch avec les deux vues monte direct à ~86 %. Mais ensuite j'ai galéré : j'ai
essayé plus d'epochs, une résolution plus grande (256 px), un nettoyage des images
ambiguës… et ça plafonnait toujours vers 0,86-0,88 sur mon indicateur clé (le rappel
Conforme à précision 95 %). Le vrai point dur, c'était la confusion **Conforme ↔
PIETRA**.

Ce qui m'a mis sur la dernière piste, c'est de raisonner sur ce que « voit » le
réseau : si un défaut n'apparaît que d'un côté, alors les features de la vue T et de la
vue B deviennent **différentes** à cet endroit. Plutôt que de juste concaténer les deux
vues, j'ai donné explicitement au modèle **l'écart absolu entre elles**, `|T − B|`, en
plus. C'est ce qui a débloqué le rappel : cette fusion `[T, B, |T−B|]` passe la cible
du cahier des charges. En ajoutant du **TTA** (moyenner la prédiction sur quelques
rotations à l'inférence), je gagne encore un peu.

Résultat final sur le jeu de test (que le modèle n'a jamais vu) : **précision 95,3 %
et rappel 88,4 % sur la classe Conforme**, pour une accuracy globale de 91,3 %. Les
deux seuils du cahier des charges sont tenus, avec de la marge. Et les erreurs qui
restent vont dans le bon sens : le modèle rejette à tort quelques bonnes châtaignes
(perte de rendement, tolérée) plutôt que de laisser passer des mauvaises.

## La mise en production

J'ai exporté le modèle en **ONNX** (avec un axe batch dynamique pour pouvoir traiter
les 24 images d'un tick d'un coup) et j'ai vérifié que les sorties ONNX collent à
celles de PyTorch. Côté latence, sur mon Mac en CPU je suis déjà autour de 48 ms pour
un tick complet ; sur la GTX 1060 en CUDA/FP16 ce sera bien plus rapide, donc le temps
réel sur 12 flux est confortable. Le modèle est minuscule (les deux branches partagent
le même backbone), il tient largement dans les 3 Go.

## Ce que je retiens

Le vrai apport de ce projet, ce n'est pas d'avoir empilé des couches, c'est d'avoir
compris le problème métier : une châtaigne se juge sur ses deux faces, et un défaut
crée une asymétrie entre les vues. C'est en modélisant ça directement (la fusion
`|T−B|`) que j'ai décroché la cible, alors que tous les réglages « génériques »
plafonnaient. Et le fait de piloter mes choix par la contrainte du cahier des charges
(précision d'abord, sur du matériel modeste) m'a évité de sur-optimiser un chiffre qui
n'aurait pas tenu en production.
