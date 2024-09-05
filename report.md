### Things to do

1. Apply changes:
    1. ~~observation vs visitation bonus~~
    2. ~~no bonus for minimum reward~~
    3. ~~KLUCB over the monitor~~
    4. ~~Per goal termination in exploration~~
2. Experiments with known monitors
3. Proofs
    1. Does MBIE(explore) visit every (observable?) state-action pairs.
    2. Does the MBIE(exploit) finds the optimal/cautious policy with there is some parts that don't have optimisim on
       them?

### Base code

branch: `mbie_episode`
commit: [c2f609d](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)

### Attempts

1. observation vs visitation bonus and no bonus for minimum reward
   - branch: `mbie_episode`
   - commit: [a8b5d2f](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
   - cluster: cedar
   - **conclusion**:

2. Per goal termination in exploration
   - branch: `mbie_episode`
   - commit: [952167c](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
   - cluster: beluga
   - **conclusion**:
   
3. KLUCB over the monitor
   - branch: `mbie_episode`
   - commit: [f6df8e7](https://github.com/alirezakazemipour/ofu/tree/mbie_episode)
   - cluster: graham
   - **conclusion**:
