# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01-crisp-finch-44/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01-crisp-finch-44/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at each case's
target, as five quantiles on the logit scale.

`node main.ts` reads every `*.in` case from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## The model

The page prints a rate, but a rate is a ratio: earners over tracked players.

    rate_i(t) = 100 * E_i(t) / T(t)

Both grow after a launch, at different speeds, and that difference is the forecast. Writing each
trophy's share as `E_i ~ T^alpha`,

    rate_i(target) = rate_i(last) * (T(target)/T(last))^(alpha - 1)

so the two things to estimate are how far `T` grows by the target and each trophy's `alpha`. Exophase
refreshes on its own schedule, so a page read between refreshes reports stale rates against a stale
count: the model fits `T ~ age^k` on every distinct count a case records, and falls back to a
launch-week prior when a history holds one unchanged page.

`alpha` is fitted per trophy where the history can measure it. Otherwise a prior keyed on how much of
the tracked population the trophy reaches stands in, calibrated on the launch-night refresh the rules
record: story trophies hold their share (alpha near 1), late and collection trophies hold almost none
(alpha near 0, since the few hundred people who had them at 638 players are the same few hundred at
14,000).

The spread is the honest one: the growth factor's error, multiplied by each trophy's exposure
`1 - alpha`, plus the trophy's own pace error. Pinball loss is minimised only by the true quantiles, so
the width is chosen to be right, not to look confident.
