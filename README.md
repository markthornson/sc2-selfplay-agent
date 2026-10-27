# sc2-selfplay-agent

En StarCraft II-agent, der lærer at spille gennem reinforcement learning og self-play, uden replays eller menneskelige data.

**Mål:** Slå den indbyggede AI på alle sværhedsgrader, med alle racer på alle slags maps, og på sigt deltage i [SC2 AI Arena](https://aiarena.net).
**Første milepæl:** Vinde stabilt mod den indbyggede AI på `VeryEasy`.

## Plan

1. **Opsætning** (dette trin): SC2, maps, Python-miljø og et Gymnasium-miljø, der kører flere SC2-spil parallelt.
2. **Første agent:** Terran på ét map, makro-handlinger, træning med PPO (med action masking) mod `VeryEasy`, indtil agenten vinder over 90 %.
3. **Self-play og stigende sværhedsgrad:** Agenten spiller mod gamle udgaver af sig selv, og vi måler løbende mod den indbyggede AI fra `Easy` og op.
4. **Udvidelse:** Flere maps, alle tre racer, finere kontrol (micro) og AI Arena-formatet.

## Opsætning på Windows

### 1. StarCraft II

Installér StarCraft II via [Battle.net](https://battle.net) (gratis Starter Edition er nok). Start spillet én gang, så det er helt opdateret, og luk det igen. python-sc2 finder selv installationen.

### 2. Maps

1. Hent map-pakken **Ladder 2019 Season 3** fra [Blizzards s2client-proto](https://github.com/Blizzard/s2client-proto#map-packs). Zip-filen er beskyttet med adgangskoden `iagreetotheeula` (du accepterer dermed Blizzards AI- og ML-licens).
2. Opret mappen `C:\Program Files (x86)\StarCraft II\Maps`, hvis den ikke findes.
3. Pak zip-filen ud, så du får fx `C:\Program Files (x86)\StarCraft II\Maps\Ladder2019Season3\AcropolisLE.SC2Map`.

Standard-mappet er `AcropolisLE`. Flere maps (også dem AI Arena bruger) kan hentes fra [AI Arenas map-side](https://aiarena.net/wiki/maps/).

### 3. Python-miljø

Brug Python 3.10–3.13. Kør i PowerShell i repo-mappen:

```powershell
.\setup.bat
```

Scriptet opretter `.venv`, installerer PyTorch med CUDA og resten af pakkerne og kører testene til sidst. Vil du gøre det i hånden, så se kommandoerne i `setup.bat`.

Scriptet skriver `CUDA: True`, når GPU'en bliver fundet. Det bruger PyTorchs CUDA 12.8-build; en anden kan vælges med miljøvariablen `TORCH_INDEX`.

### 4. Smoke-test

```powershell
.\smoke.bat --realtime         # ét spil mod VeryEasy i normal hastighed, så du kan se med
.\smoke.bat                    # samme spil i fuld fart
.\smoke.bat --policy random
```

Den scriptede politik bygger marines og angriber, og den bør vinde mod `VeryEasy`. Det viser, at handlingsrummet kan vinde, før vi begynder at træne. Replays gemmes i `replays/`.

## Træning

```powershell
.\train.bat                     # 6 parallelle spil mod VeryEasy, 2 mio. beslutninger
.\train.bat --n-envs 8          # flere spil, hvis CPU og RAM kan følge med
.\tensorboard.bat               # følg med på http://localhost:6006 (i et andet vindue)
```

Træningen skriver løbende sejrsprocenten for de seneste 100 spil, og i TensorBoard ligger den under `games/win_rate`. Checkpoints gemmes i `checkpoints/<kørsel>/` for hver 50.000 beslutninger. Ctrl+C stopper og gemmer `interrupted.zip`. En kørsel kan fortsættes med `--resume checkpoints/<kørsel>/<fil>.zip`.

Et SC2-spil, der går ned eller hænger, bliver genstartet automatisk, så en lang kørsel ikke stopper. Sådanne spil tælles i `games/errors`.

Mål sejrsprocenten for en model:

```powershell
.\evaluate.bat checkpoints\<kørsel>\final.zip --games 20
.\evaluate.bat checkpoints\<kørsel>\final.zip --difficulty Easy
```

## Sådan virker det

| Fil | Indhold |
| --- | --- |
| `sc2agent/env.py` | `SC2Env` (Gymnasium) og `SC2VecEnv` (Stable-Baselines3), der kører hvert spil i sin egen proces |
| `sc2agent/bot.py` | Botten inde i spillet. Sender observationer, modtager handlinger og klarer det, politikken ikke bestemmer (minedrift, hærbevægelse) |
| `sc2agent/actions.py` | De 11 makro-handlinger og action masks |
| `sc2agent/features.py` | Observationen: 27 normaliserede tal om økonomi, hær, bygninger og fjenden |
| `sc2agent/config.py` | Indstillinger for et spil (map, modstander, sværhedsgrad, tidsgrænse, reward shaping) |

**Handlinger:** NOOP, træn SCV, byg supply depot, byg barracks, træn marines, byg refinery, ekspandér, byg engineering bay, opgradér infantry weapons, angrib, forsvar. Politikken vælger én handling hvert 16. game loop (ca. 0,7 sekunder spiltid). Ugyldige handlinger maskeres væk.

**Reward:** +1 for sejr, -1 for nederlag, 0 for uafgjort (tidsgrænse). Dertil en lille belønning for at ødelægge fjendens enheder og bygninger (`shaping_scale`), så agenten får signal, før den vinder sit første spil.

**Hvorfor én proces pr. spil:** python-sc2 skal køre i hovedtråden i sin proces og har en fælles kill switch pr. proces. Derfor har hvert miljø sin egen worker-proces, og `SC2VecEnv` sender handlinger til alle spil på én gang og samler svarene ind bagefter.

**Ressourcer:** Hver SC2-instans bruger cirka én CPU-kerne og 1–2 GB RAM. Med en Core Ultra 7 265K kan 6–8 parallelle spil være et godt udgangspunkt i trin 2.
