### Things to do

1. Apply changes:
    1. ~~observation vs visitation bonus~~
    2. ~~no bonus for minimum reward~~
    3. ~~KLUCB over the monitor~~
    4. ~~Per goal termination in exploration~~
    5. ~~Natural environment termination during exploration~~
2. ~~Experiments with known monitors~~
3. Proofs
    1. Does MBIE(explore) visit every (observable?) state-action pairs.
    2. Does the MBIE(exploit) finds the optimal/cautious policy with there is some parts that don't have optimisim on
       them?
4. Posting the slides

### Base code

branch: `mbie_episode`
commit: [c2f609d](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)

### Attempts

1. observation vs visitation bonus and no bonus for minimum reward
    - branch: `mbie_episode`
    - commit: [a8b5d2f](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
    - cluster: cedar
    - **conclusion**: It's fine. Except in Distract + Button 10% that could be corrected by lowering beta.

2. Per goal termination in exploration
    - branch: `mbie_episode`
    - commit: [952167c](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
    - cluster: beluga
    - **conclusion**: Was much worse than mine because it lingers on every state-action pairs in partially observable
      cases. While mine plan the most collectively. Maybe it's better to leave it off to the planner and use the natural
      termination. Edit: Nope, mine is the right way!

3. KLUCB over the monitor
    - branch: `mbie_episode`
    - commit: [f6df8e7](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
    - cluster: graham
    - **conclusion**: I need to ask Mike about it. Results are slower but much more on target. interesting! But it seems
      to be the right approach; consider a T-Maze that the agent always start the interaction with the button off and
      the button is outside the hallway! Dependency on only environment pairs makes the hallway cells mistakenly
      never-observable!

4. Experiments with known monitors
    - branch: `mbie_episode_known_monitor`
    - commit: [35a0f42](https://github.com/alirezakazemipour/ofu/tree/mbie_episode_known_monitor)
    - cluster: narval
    - **conclusion**: Results are slower which is incompatible with the expectations! But, because the other error-prone
      things, it's not reliable.

5. Natural environment termination during exploration
    - branch: N/A
    - commit: N/A
    - cluster: cedar
    - **conclusion**: does not work! Use mine!

6. Combination of above changes. (unknown monitor)
    - branch: `mbie_episode`
    - commit: [a222d5c](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
    - cluster: graham
    - **conclusion**:
