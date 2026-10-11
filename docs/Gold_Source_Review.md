# Gold-answer source review

Reference answers and query texts are preserved from the supplied benchmark. The canonical CSV adds `supplied_verification_status` to retain the original primary/secondary verification claims. The active `verification_status` does not certify independent factual checks. The source audit contains all 120 U items and distinguishes six explicit review concerns from the remaining unverified supplied claims.

| Query | Review concern |
|---|---|
| TN-U019 | Recorded scoring notes state that the 4.6 double-faults-per-match reference is inconsistent with match statistics. Resolve from match-level data. |
| F1-U010 | The linked official article concerns one Baku Sprint's total positions gained, not season-wide lap-one gains. |
| F1-U008 | The linked DHL page shows 2023 award points, while average-time data shown are for 2024. A 2023 average and population/exclusion rules are needed. |
| F1-U007 | A full-season driver-standings page does not establish points immediately after the Bahrain GP. Use that race-stage snapshot. |
| TN-U004 | The supplied historical match-statistics endpoint could not be retrieved during source triage. Check the 67% target from a valid archive. |
| TN-U008 | The supplied match-statistics endpoint needs an archived replacement and verification of the 7-of-8 target. |

The six concerns are not a complete factual audit or proof that every listed answer is wrong. No replacement number was invented. Official source checks for F1-U010 and F1-U008:

- https://www.formula1.com/en/latest/article/f1-sprint-who-gained-and-lost-the-most-positions-in-the-baku-sprint.6DyfFgggXa0YBZ6qtloJFJ
- https://inmotion.dhl/en/formula-1/fastest-pit-stop-award-2024

The first covers the single Baku Sprint. The second contains 2023 award standings by points and a 2024 averages section. An award-points leader does not by itself establish a season-average statistic.

For each derived target, preserve the actual event/time snapshot, source data, denominator, exclusions, units, rounding, and calculation. Update source status only after reviewing that evidence. Correcting a gold answer requires rerating affected responses; changing a query may require new inference. Version such scientific changes separately from this packaging repair.
