# Input data and preprocessing

## Required files

Provide UTF-8 CSV files with headers in the configured input directory. Preserve
column order because the loaders access several fields by position. Synthetic
datasets can use any consistent city identifiers and cultural labels.

| File | Columns in order | Meaning |
|---|---|---|
| city_pop.csv | city_id, city_pop, culture_type, removed_pop | City identifier, total population, native culture, removed population |
| city_culture_pop.csv | CityID, followed by state-code columns | Population in each cultural compartment |
| city_pop_flow.csv | o_city_id, d_city_id, flow_pop | Origin, destination, annual movement count |
| cultural_strength.csv | culture1, culture2, cultural_strength | Source culture, target culture, cultural antibody CA |

Use nonnegative integer population and flow counts in consistent population units.
Document any scaling factor. Each city identifier must be unique in the population
and initial-state tables. All flow endpoints must occur in both tables.

## Cultural states

`N_L_0` and `H_L_0` represent single-culture native and heterogeneous states.
`I_B_D` and `A_B_D` represent integration and acculturation states carrying B and D.
`R_0_0` represents the removed compartment, excluded from acculturation.
Use a consistent ordering for two-culture labels. The loaders generate state
combinations from the cultural labels in the city population table; match the
corresponding state-column names in the initial-state table. Supply columns for
the required generated states, using zero counts for unoccupied compartments.

Sum mutually exclusive compartments once to obtain each city's total population.
The R count must match removed_pop. A two-gene compartment contributes to each of
its two cultural-label counts; R_0_0 contributes no culture-specific label.

## Cultural antibody

The coefficient table supplies CA for ordered culture pairs. Conversion terms
are divided by CA. For CA=1/lambda and similarity lambda in (0,1], use CA>=1.
Positive infinity (`inf`) represents lambda=0 and produces zero similarity-mediated
transfer. Missing directed pairs default to CA=1. Provide all required directed
pairs explicitly to control cultural interactions. Values must be positive and
must not be NaN.

## Constructing synthetic inputs

1. Choose cities, native culture labels, population totals, and removed counts.
2. Allocate initial populations to N, H, I, A and R compartments. Ensure each
   city state total equals city_pop and its R count equals removed_pop.
3. Specify annual directed flows between the chosen cities. Resolve duplicate
   OD keys before loading and keep outgoing totals compatible with city populations.
4. Define cultural similarities and convert them to CA, recording the treatment
   of zero similarity and the direction convention.
5. Save the four tables in the documented order and run validate_inputs.py.
6. Retain the initial tables, configuration and random seeds for each run.

## Preparing region-based research inputs

Harmonize administrative units, city codes, reference years and population units
across the input sources. For dialect and cuisine profiles, record the dialect-code
aggregation level, county weighting, restaurant classification, deduplication and
normalization. Document the regionalization method and any manual spatial review.
Record how population and migration information are converted into initial
cultural compartments. Aggregate OD observations to the required annual time step
and resolve missing endpoints, duplicate pairs, self-flows and excessive outflows.
The loader retains the last value for a duplicate OD key. Keep processed inputs
fixed during a run; run metadata stores their hashes.
