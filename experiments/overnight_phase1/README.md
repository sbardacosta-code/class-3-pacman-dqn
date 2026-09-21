# Overnight experiment: preserved first stage

This is the completed first stage of the [overnight continuation](../../README.md). Its selected model is an intermediate result. The continuation recipe was fixed before these classroom scores and does not use them to choose training settings.

Run `20260921_005003_092029`: status `time_budget`, 1,975 completed games, 1,535,392 decisions, 381,349 recorded updates, and 7200.011 training seconds.

[Executed notebook](pacman_dqn.ipynb) · [Config](results/config.json) · [Training CSV](results/training.csv) · [Summary](results/training_summary.json) · [Comparison](results/comparison.json) · [Selection](results/model_selection.json) · [Initial plan](experiment_plan.md) · [Overnight extension](overnight_plan.md)

| Seed | Fresh baseline | Stage-one selected |
| --- | ---: | ---: |
| 101 | 350 | 2500 |
| 202 | 500 | 2120 |
| 303 | 320 | 2490 |
| 404 | 800 | 2520 |
| 505 | 490 | 1130 |
| Mean | 492 | 2152 |

![Training dashboard](results/training_dashboard.png)

The GIFs show at most the first 20 seconds. The final GIF is the best of five games; it does not demonstrate typical performance. The full stage-one ZIP and all large checkpoints remain local under `pacman_runs/`.

![Fresh baseline](results/demos/episode_0000.gif)

![Stage-one best selected game](results/demos/final_best.gif)

<details><summary>Every stage-one intermediate gameplay GIF</summary>

![episode_0025](results/demos/episode_0025.gif)

![episode_0050](results/demos/episode_0050.gif)

![episode_0075](results/demos/episode_0075.gif)

![episode_0100](results/demos/episode_0100.gif)

![episode_0125](results/demos/episode_0125.gif)

![episode_0150](results/demos/episode_0150.gif)

![episode_0175](results/demos/episode_0175.gif)

![episode_0200](results/demos/episode_0200.gif)

![episode_0225](results/demos/episode_0225.gif)

![episode_0250](results/demos/episode_0250.gif)

![episode_0275](results/demos/episode_0275.gif)

![episode_0300](results/demos/episode_0300.gif)

![episode_0325](results/demos/episode_0325.gif)

![episode_0350](results/demos/episode_0350.gif)

![episode_0375](results/demos/episode_0375.gif)

![episode_0400](results/demos/episode_0400.gif)

![episode_0425](results/demos/episode_0425.gif)

![episode_0450](results/demos/episode_0450.gif)

![episode_0475](results/demos/episode_0475.gif)

![episode_0500](results/demos/episode_0500.gif)

![episode_0525](results/demos/episode_0525.gif)

![episode_0550](results/demos/episode_0550.gif)

![episode_0575](results/demos/episode_0575.gif)

![episode_0600](results/demos/episode_0600.gif)

![episode_0625](results/demos/episode_0625.gif)

![episode_0650](results/demos/episode_0650.gif)

![episode_0675](results/demos/episode_0675.gif)

![episode_0700](results/demos/episode_0700.gif)

![episode_0725](results/demos/episode_0725.gif)

![episode_0750](results/demos/episode_0750.gif)

![episode_0775](results/demos/episode_0775.gif)

![episode_0800](results/demos/episode_0800.gif)

![episode_0825](results/demos/episode_0825.gif)

![episode_0850](results/demos/episode_0850.gif)

![episode_0875](results/demos/episode_0875.gif)

![episode_0900](results/demos/episode_0900.gif)

![episode_0925](results/demos/episode_0925.gif)

![episode_0950](results/demos/episode_0950.gif)

![episode_0975](results/demos/episode_0975.gif)

![episode_1000](results/demos/episode_1000.gif)

![episode_1025](results/demos/episode_1025.gif)

![episode_1050](results/demos/episode_1050.gif)

![episode_1075](results/demos/episode_1075.gif)

![episode_1100](results/demos/episode_1100.gif)

![episode_1125](results/demos/episode_1125.gif)

![episode_1150](results/demos/episode_1150.gif)

![episode_1175](results/demos/episode_1175.gif)

![episode_1200](results/demos/episode_1200.gif)

![episode_1225](results/demos/episode_1225.gif)

![episode_1250](results/demos/episode_1250.gif)

![episode_1275](results/demos/episode_1275.gif)

![episode_1300](results/demos/episode_1300.gif)

![episode_1325](results/demos/episode_1325.gif)

![episode_1350](results/demos/episode_1350.gif)

![episode_1375](results/demos/episode_1375.gif)

![episode_1400](results/demos/episode_1400.gif)

![episode_1425](results/demos/episode_1425.gif)

![episode_1450](results/demos/episode_1450.gif)

![episode_1475](results/demos/episode_1475.gif)

![episode_1500](results/demos/episode_1500.gif)

![episode_1525](results/demos/episode_1525.gif)

![episode_1550](results/demos/episode_1550.gif)

![episode_1575](results/demos/episode_1575.gif)

![episode_1600](results/demos/episode_1600.gif)

![episode_1625](results/demos/episode_1625.gif)

![episode_1650](results/demos/episode_1650.gif)

![episode_1675](results/demos/episode_1675.gif)

![episode_1700](results/demos/episode_1700.gif)

![episode_1725](results/demos/episode_1725.gif)

![episode_1750](results/demos/episode_1750.gif)

![episode_1775](results/demos/episode_1775.gif)

![episode_1800](results/demos/episode_1800.gif)

![episode_1825](results/demos/episode_1825.gif)

![episode_1850](results/demos/episode_1850.gif)

![episode_1875](results/demos/episode_1875.gif)

![episode_1900](results/demos/episode_1900.gif)

![episode_1925](results/demos/episode_1925.gif)

![episode_1950](results/demos/episode_1950.gif)

![episode_1975](results/demos/episode_1975.gif)

</details>
