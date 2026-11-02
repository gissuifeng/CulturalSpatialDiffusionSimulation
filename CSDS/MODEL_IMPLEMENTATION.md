# Model implementation and parameters

## Parameter settings

| Configuration field | Symbol | Default | Role |
|---|---|---|---|
| contact_rate | r | 0.002 | Common contact-mediated conversion coefficient |
| transform_rate_H | mu_H | 0.30 | N conversion mediated by H |
| transform_rate_I | mu_I | 0.20 | N conversion mediated by I |
| transform_rate_A | mu_A | 0.15 | N conversion mediated by A |
| htoI_rate | gamma | 0.20 | H-to-I conversion coefficient |
| itoA_rate | theta | 0.15 | I-to-A conversion coefficient |
| simulation.iterations | T | 10 | Number of annual updates |

Defaults define a reference scenario. The configuration controls the duration
and coefficient values. `--seed` initializes Python and NumPy random generators;
PYTHONHASHSEED controls cultural category ordering in multinomial mode.

## Annual update sequence

1. Process origin cities in input order. Sample outgoing cultural compartments
   and destinations, immediately subtracting and adding migrants. Cities processed
   later in the update can redistribute arrivals from earlier origins.
2. Assign single-gene N/H status according to the destination city's native culture.
3. Calculate within-city conversions from post-migration compartment counts.
4. Apply accumulated compartment changes and update city totals.
5. Record city states and cultural compositions of directed migration flows.

The resulting state is carried into the next update. The annual OD input is reused
at each update. Births, deaths, aging and marriage are outside the state transitions.

## Cultural conversion

Native contact with H, I or A contributes to I. Contact terms scale with native
population, source compartment share, r and the corresponding mu coefficient,
and are divided by CA. H-to-I conversion is gamma*H/CA; I-to-A conversion is
theta*I/CA. The same I-to-A transfer is subtracted from I and added to A.

CA is selected for the nonnative-to-native culture pair. If an I/A compartment
contains two nonnative labels, one label is randomly selected to form the pair.
Positive infinity yields zero transfer. The coefficient validation rejects
nonpositive and NaN values.

## Sampling and numerical behavior

Multinomial mode samples migration counts through multinomial draws and converts
accumulated changes with int(change), truncating toward zero. Roulette mode draws
migrants individually and uses int(round(change)), with ties rounded to even.
The modes use different sampling and rounding schemes, so the same random seed
can produce different trajectories.

Migration probabilities are calculated using nonnegative clipped compartment
counts. Stored negative counts are preserved. Multinomial draws use replacement
probabilities and can exceed a source compartment count. Independent transfer
rounding can affect population conservation. Inspect compartment nonnegativity
and population balance when selecting inputs and interpreting outputs.

## Output indicators

Total model population sums mutually exclusive compartments, including R, once.
A culture's carrier count sums compartments bearing its label. A two-gene
compartment contributes once to each corresponding label; R_0_0 contributes no
culture-specific count. Total cultural-label count sums all cultural carrier counts.
Normalized cultural shares divide a culture's count by total cultural-label count
and sum to one. I/A state proportions divide their compartment populations by
total model population.

The relative growth rate of a cultural share is (p_end-p_start)/p_start. State
the selected time endpoints and percentage convention. An initial share of zero
makes this relative growth undefined and requires an explicit analysis policy.
A decreasing cultural share denotes reduced relative representation and can
coexist with an increasing carrier count.
