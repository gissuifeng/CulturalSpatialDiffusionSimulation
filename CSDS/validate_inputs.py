"""Read-only checks for CSDS processed CSV inputs; no files are modified."""
import argparse
import csv
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data/source_data')
    args = parser.parse_args()
    tables = {}
    errors = []
    required = {
        'city_pop.csv': ['city_id', 'city_pop', 'culture_type', 'removed_pop'],
        'city_pop_flow.csv': ['o_city_id', 'd_city_id', 'flow_pop'],
        'cultural_strength.csv': ['culture1', 'culture2', 'cultural_strength'],
        'city_culture_pop.csv': ['CityID'],
    }
    for name, expected in required.items():
        try:
            with (args.data / name).open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if (reader.fieldnames or [])[:len(expected)] != expected:
                    errors.append(name + ': unexpected header order')
                tables[name] = list(reader)
        except (OSError, UnicodeError) as exc:
            errors.append(str(exc))
    if errors:
        for e in errors: print('ERROR:', e)
        return 1
    city = tables['city_pop.csv']; ids = {r['city_id'] for r in city}
    if len(ids) != len(city): errors.append('Duplicate city IDs')
    states = tables['city_culture_pop.csv']
    stateids = {r['CityID'] for r in states}
    if len(stateids) != len(states): errors.append('Duplicate state-row IDs')
    if stateids != ids: errors.append('Population and state city ID sets differ')
    pop = {r['city_id']: r for r in city}
    mismatches = []
    try:
        for r in city:
            for field in ['city_pop', 'removed_pop']:
                v = float(r[field])
                if not math.isfinite(v) or v < 0 or v != int(v):
                    errors.append('Invalid population count: ' + r['city_id'])
        for r in states:
            vals = [float(v) for k,v in r.items() if k != 'CityID']
            if any(not math.isfinite(v) or v < 0 or v != int(v) for v in vals):
                errors.append('Invalid state count: ' + r['CityID'])
            if r['CityID'] in pop:
                source = pop[r['CityID']]
                if sum(vals) != float(source['city_pop']) or float(r.get('R_0_0',0)) != float(source['removed_pop']):
                    mismatches.append(r['CityID'])
        if mismatches: errors.append('State/total or removed-count mismatches: ' + str(len(mismatches)) + ' cities; first 10: ' + str(mismatches[:10]))
        flows = tables['city_pop_flow.csv']
        unknown = sorted({r[k] for r in flows for k in ['o_city_id','d_city_id']} - ids)
        if unknown: errors.append('Unknown OD endpoints: ' + str(unknown))
        pairs = [(r['o_city_id'],r['d_city_id']) for r in flows]
        if len(set(pairs)) != len(pairs): errors.append('Duplicate OD pairs')
        for r in flows:
            v = float(r['flow_pop'])
            if not math.isfinite(v) or v < 0 or v != int(v):
                errors.append('Invalid flow count'); break
        strength = tables['cultural_strength.csv']
        pairs = [(r['culture1'],r['culture2']) for r in strength]
        if len(set(pairs)) != len(pairs): errors.append('Duplicate strength pairs')
        if any(math.isnan(float(r['cultural_strength'])) or float(r['cultural_strength']) <= 0 for r in strength):
            errors.append('CA must be positive; positive infinity represents zero similarity')
    except (ValueError,TypeError,KeyError) as exc:
        errors.append('Malformed table value: ' + str(exc))
    for name, rows in tables.items(): print(name + ': ' + str(len(rows)) + ' rows')
    for e in errors: print('ERROR:', e)
    print('Input checks failed.' if errors else 'Input checks passed.')
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
