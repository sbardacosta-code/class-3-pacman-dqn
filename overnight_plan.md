# Overnight extension of the Ms. Pac-Man experiment

The student made approximately **six more hours** available at **2026-09-21 08:07 UTC (01:07 PDT)**. This replaces the earlier three-hour overall constraint in [the initial plan](experiment_plan.md). The current two-hour stage stays unchanged. A second stage continues its saved learner until a planned training cutoff of **13:37 UTC (06:37 PDT)**, leaving approximately 30 minutes for final evaluation, saving, review, and publication by **14:07 UTC (07:07 PDT)**.

The extension was requested before the first 250-game validation result arrived. That result was essentially flat: the inherited model averaged 1,199 on the ten development games, while the 250-game candidate averaged 1,194. The existing selection rule retained the inherited model. This is an observation, not evidence that longer training will necessarily succeed.

## Training choices

Keep exploration **0.10**, learning rate **0.00005**, three-step returns, prioritized replay, and the small life-loss penalty unchanged. Raise the combined advanced experiment's ceiling from 2,500 to **6,000 game starts**, with time expected to bind first. The original inherited 1,069 games remain separately disclosed. There is no automatic further extension after the overnight deadline.

After stage one finishes and its executed notebook and full ZIP are saved, restore the **last learner** from its full snapshot: online network, target network, Adam optimizer, replay contents and priorities, and saved random states. Continue the aggregate decision/update counters so replay filling, random warm-up, importance correction, and update/target schedules do not restart. The selected playback checkpoint can differ from the last learner; preserve the strongest selection checkpoint as an incumbent rather than substituting it for the continuing learner.

The snapshot does not include emulator state. Start fresh games after the previous **started-game** counter, abandon the recorded partial games, and clear only their pending multi-step tails. Keep the already emitted replay transitions. Record every abandoned game and decision and the number of discarded pending items. This is continuation of the learning state, not exact mid-game resumption. Completed CSV decisions, abandoned partial-game decisions, and final active-game decisions must reconcile to the combined total.

The final notebook will show a genuine untrained baseline, the original inherited model's separately labeled evaluation, an explicit restoration of the stage-one learner, all saved intermediate gameplay, continued training, and final evaluation. Its configuration and provenance will identify the source snapshot and a verified public resume bundle. The original stage-one executed notebook and evidence are preserved in `experiments/overnight_phase1/`.

## Selection and final tests

Keep the existing ten development seeds **20001–20010** and criterion **mean raw score − 0.5 × population standard deviation**. Carry the full selection history and incumbent into stage two. Evaluate candidates at each combined 250-game milestone and at each stage's end; retain the earlier candidate on ties. Stage one's final five classroom scores are recorded automatically by its existing notebook, but will not be used to change the continuation recipe or select a model.

The classroom evaluation stays at seeds **101, 202, 303, 404, 505**, **5% exploration**, and **3,000 decisions per game**, with unchanged environment, preprocessing, actions, and network architecture. After the final selection, report all five scores and their mean. Only then run the predeclared twenty additional seeds **30001–30020** for the original inherited model and final selected model. These extra games perform no learning or further selection.

The target remains **classroom mean at least 3,000 and at least 18/20 additional games at least 3,000**. More time is an opportunity for improvement, not a guarantee. Report a regression, an inherited-model winner, missed target, or interrupted run honestly. Preserve the distinction between all training attempted and the budget that produced the selected checkpoint.

## Operation and evidence

Use the tested Apple MPS setup with four training environments and one shared learner. Keep the Mac on power with its lid open. A temporary sleep assertion lasts only while the training supervisor runs; no permanent power settings change. Save the notebook outputs regularly, gameplay/checkpoints every 25 completed games, all training/evaluation records, and full local ZIPs. The final notebook's copied stage-one GIFs are labeled as restored evidence rather than newly generated training.

The supervisor starts stage two only after stage one's notebook executor reports successful completion and its ZIP is closed. It derives the second stage's relative time allowance from the remaining absolute overnight deadline, reserving startup time. A periodic follow-up checks for completion or actionable failures; it must not start duplicate training or add another run past the deadline.

Publication includes the executed final notebook, README, plot, all GIFs, all numerical evidence, and this amended plan. Large checkpoints stay in the local ZIPs or a GitHub release. A resume bundle published with its SHA-256 provides the exact starting learner state for reproducing stage two. All claims about gameplay will be based on the actual saved previews and scores.
