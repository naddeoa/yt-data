
# [DONE] Phase1
- `pkmn_oreily_64f-8x_60dim-normal_410batch-all`

# [INPROGRESS] Phase2
- Frozen generator from `pkmn_oreily_64f-8x_60dim-normal_410batch-all` (Phase1)
- Discriminator from scratch for 10,000 epohcs

# Phase3

## Phase3 (option):
- Discriminator from `pkmn_pretrained-gen_64f-16x_60dim-normal_410batch-all` (Phase2)
- Higher capacity generator from scratch

## Phase3 (option):
- Discriminator from `pkmn_pretrained-gen_64f-16x_60dim-normal_410batch-all` (Phase2)
- Generator `pkmn_oreily_64f-8x_60dim-normal_410batch-all` (Phase1 and Phase2)


## Phase3 (option):
- Continue Phase2 for 100,000 epochs



## [FAIL] Phase3:
Immediately failed.
- Frozen discriminator from `pkmn_pretrained-gen_64f-16x_60dim-normal_410batch-all` (Phase2)
- Generator `pkmn_oreily_64f-8x_60dim-normal_410batch-all` (Phase1 and Phase2)
