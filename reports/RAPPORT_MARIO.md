# CastagNet — ma démarche

*Note de méthode : pour ce projet je me suis appuyé sur une IA générative (Claude)
comme copilote — pour auditer le dataset, retrouver les bons repos de référence,
écrire le pipeline et débugger. Les décisions et les raisons sont les miennes ;
l'IA a surtout accéléré la recherche et la mise en œuvre.*

---

## Le point de départ

Une châtaigne est filmée par deux caméras, une par-dessus (T) et une par-dessous (B) —
c'est le principe qu'on nous a présenté et qui est rappelé dans le README de
labellisation. Tout mon travail part de là : comment exploiter concrètement ces deux
vues, d'abord dans la donnée, puis dans le modèle.

## Ce que j'ai trouvé en creusant la donnée

En regardant les labels de près, je suis tombé sur un décalage important. Il y a deux
colonnes : le label déduit du nom de fichier, et la vraie cible relue. En croisant les
deux, j'ai vu que **35,6 % des images avaient été reclassées en « Vide »** à la
relecture — un créneau photographié peut être vide. Le rapport statistique de départ,
lui, ne connaissait que 3 classes et ignorait complètement « Vide » : il était périmé.
Je l'ai donc refait sur les vrais labels. J'ai aussi terminé la relecture qui restait,
et j'ai remarqué au passage que ce lot était mal étiqueté à l'origine (à peine 5 % de
vrais PIETRA dedans) — un bon exemple de l'hétérogénéité du diagnostic.

## Relier les deux vues, et décider quoi en faire

Puisqu'un fruit = deux photos, la vraie question c'était : comment les relier, et quel
label donner à la paire. Pour l'appariement, j'ai utilisé une clé simple : le nom de
fichier privé de la position `T/B`. Ça reconstitue 19 078 châtaignes (16 176 paires
complètes).

Ensuite j'ai regardé les cas où les deux vues n'étaient pas d'accord, et là il y a un
résultat net : **100 % de ces désaccords viennent d'une face vue comme « Vide »** —
jamais un vrai conflit entre deux classes. Autrement dit, quand le fruit est là, les
deux caméras sont d'accord ; la seule « divergence » c'est quand une caméra ne le
capte pas (le fruit est hors de son champ). D'où ma règle d'agrégation : **la classe
non-Vide gagne**, et en cas de vrai doute je prends la plus sévère. Ce choix colle au
cahier des charges : on préfère rejeter une bonne châtaigne (perte de rendement,
tolérée) que laisser passer une mauvaise (qui dégrade le lot). C'est aussi ce qui m'a
convaincu qu'il fallait vraiment décider **sur la paire**, pas sur une image seule :
environ un tiers des fruits présents apparaissent « Vide » sur une seule de leurs deux
vues.

## Comment j'ai monté le pipeline, et pourquoi ces choix

J'ai piloté mes choix par la contrainte de production : le modèle doit tourner sur une
vieille GTX 1060 de 3 Go et traiter 12 flux en temps réel. Pour ne pas partir de zéro,
j'ai regardé ce que font les projets sérieux du domaine, et voilà ce qui m'a orienté :

- **Architecture 2 vues** — des travaux sur le tri de fruits multi-caméra (un papier de
  grading de pommes multi-view, et SeedSortNet sur le tri de graines) m'ont conforté
  dans l'idée d'un réseau **dual-branch** : deux extracteurs de features partagés sur T
  et B, puis une fusion.
- **Backbone léger** — j'ai pris **MobileNetV3** (via `torchvision`), parce que c'est
  rapide et petit : parfait pour la 1060 et pour empiler les 24 images d'un tick (2
  vues × 12 flux) en un seul passage.
- **Prétraitement** — l'info utile est dans le disque central du slot, donc je recadre
  au centre et j'applique un **masque circulaire** (OpenCV) pour virer le fond. Et
  comme le disque est invariant en rotation, j'augmente en tournant les images de 0 à
  360° — une augmentation quasi gratuite.
- **Découpage train/val/test** — le piège classique quand un objet a plusieurs images,
  c'est la fuite : il ne faut jamais que la vue T d'un fruit soit en train et sa vue B
  en test, sinon le modèle « reconnaît le fruit » et les scores sont faussés. J'ai donc
  groupé par châtaigne avec `StratifiedGroupKFold` (scikit-learn), en stratifiant aussi
  par classe et par année. Vérifié : zéro fuite.
- **Structure de projet et suivi** — je me suis inspiré des templates PyTorch de
  référence pour organiser le code proprement, tout en gardant une boucle
  d'entraînement « nue » (sans framework qui masque la mécanique), et je trace tous mes
  runs dans **MLflow** pour comparer les modèles à égalité.

J'ai aussi récupéré des images depuis les vidéos de la machine : découpage en frames,
détection du créneau circulaire, crop autour de chaque châtaigne — de quoi enrichir le
dataset au même format que les images existantes.

## L'itération, et ce qui a fini par débloquer

Ma baseline (un petit CNN codé à la main sur une seule image) plafonne vers 66 % et
rate presque totalement PIETRA — logique, un défaut ne se voit souvent que d'un côté.
Le dual-branch avec les deux vues monte direct à ~86 %. Mais ensuite j'ai galéré :
plus d'epochs, une résolution plus grande, un nettoyage des images ambiguës… et ça
plafonnait toujours vers 0,86-0,88 sur mon indicateur clé (le rappel Conforme à
précision 95 %). Le vrai point dur, c'était la confusion **Conforme ↔ PIETRA**.

Ce qui m'a mis sur la dernière piste, c'est de raisonner sur ce que « voit » le réseau :
si un défaut n'apparaît que d'un côté, alors les features de la vue T et de la vue B
deviennent **différentes** à cet endroit. Plutôt que de juste concaténer les deux vues,
j'ai donné explicitement au modèle **l'écart absolu entre elles**, `|T − B|`, en plus.
C'est ce qui a débloqué le rappel : la fusion `[T, B, |T−B|]` passe la cible du cahier
des charges. En ajoutant du **TTA** (moyenner la prédiction sur quelques rotations à
l'inférence), je gagne encore un peu.

Résultat sur le jeu de test (jamais vu à l'entraînement) : **précision 95,3 % et rappel
88,4 % sur la classe Conforme**, pour une accuracy globale de 91,3 %. Les deux seuils du
cahier des charges sont tenus, avec de la marge. Et les erreurs restantes vont dans le
bon sens : le modèle recale à tort quelques bonnes châtaignes plutôt que de laisser
passer des mauvaises.

Ce modèle-là me paraissait déjà solide, mais pour ne pas me fier à un seul découpage
j'ai voulu vérifier qu'il était **stable**. J'ai donc fait une validation croisée en 5
découpages (toujours groupés par châtaigne pour éviter la fuite) : je retombe sur une
précision de 94,9 % ± 1,2 et un rappel de 86,8 % ± 1,9. L'écart-type de 1 à 2 points me
confirme que le résultat n'est pas un coup de chance. Ces modèles-là m'ont juste servi à
tester la robustesse — le modèle que je garde reste celui d'avant, entraîné plus
longtemps et avec le TTA, qui a la petite marge en plus.

## En deux mots, comment il marche

Concrètement, pour une châtaigne je passe ses deux photos (dessus et dessous) dans le
même extracteur, ce qui me donne deux vecteurs de caractéristiques. Je les combine en
gardant les deux **plus leur écart** `|T−B|`, et une petite couche finale sort les 4
classes. À l'usage, je moyenne la prédiction sur quelques rotations (le TTA) et je
n'accepte « Conforme » que si le modèle est assez sûr — c'est ce seuil qui me garantit la
précision. En production, un cycle traite les 12 caméras d'un coup, soit 6 paires, en un
seul passage.

## La mise en production

J'ai exporté le modèle en **ONNX**, avec un axe batch dynamique pour pouvoir traiter les
24 images d'un tick d'un coup, et j'ai vérifié que les sorties ONNX collent à celles de
PyTorch. Côté latence, sur mon Mac en CPU je suis déjà autour de 48 ms pour un tick
complet ; sur la GTX 1060 en CUDA/FP16 ce sera nettement plus rapide, donc le temps réel
sur 12 flux est confortable. Le modèle est minuscule (les deux branches partagent le
même backbone), il tient largement dans les 3 Go.

## Ce que je retiens

L'apport de ce projet, ce n'est pas d'avoir empilé des couches, c'est d'avoir traduit le
problème métier en modèle : une châtaigne se juge sur ses deux faces, et un défaut crée
une asymétrie entre les vues. C'est en modélisant ça directement (la fusion `|T−B|`) que
j'ai décroché la cible, alors que tous les réglages génériques plafonnaient. Et le fait
de piloter mes choix par la contrainte du cahier des charges — la précision d'abord, sur
du matériel modeste — m'a évité d'optimiser un chiffre qui n'aurait pas tenu en
production.
