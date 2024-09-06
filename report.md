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
      cases. While mine plan the most collectively.

3. KLUCB over the monitor
    - branch: `mbie_episode`
    - commit: [f6df8e7](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
    - cluster: graham
    - **conclusion**: I need to ask Mike about it. Results are slower but much ore on target.

4. Experiments with known monitors
    - branch: `mbie_episode_known_monitor`
    - commit: [35a0f42](https://github.com/alirezakazemipour/ofu/tree/mbie_episode_known_monitor)
    - cluster: narval
    - **conclusion**:
   
5. Natural environment termination during exploration
6. - branch: N/A
    - commit: N/A
    - cluster: cedar
    - **conclusion**:
