"""Full grid search over every model, both cities. Saves results + fitted pipelines."""
import warnings, logging, json, time
warnings.filterwarnings('ignore'); logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
import pandas as pd
from DengAI import load_config, load_raw, split_cities, build_pipelines
from DengAI import evaluate as ev

cfg = load_config()
tr, lb, te = load_raw()
data = ev.city_data(split_cities(tr, lb, te), cfg)
names = list(cfg.models.models.keys())
print('models:', names, flush=True)

t0 = time.time()
pipes = build_pipelines(cfg, names=names)
searches, best, cv_results = ev.run_searches(pipes, data, cfg, names=names)
hold, _ = ev.holdout_table(best, data)
results = cv_results.merge(hold, on=['City', 'Model'])
print('\n' + results.round(3).to_string(index=False), flush=True)

final = ev.refit_final(best, data)
written = ev.save_pipelines(final, cfg, names=names)
sub = ev.make_submission(final, data, cfg, path='submission_dengai.csv')

results.to_csv('results_full_search.csv', index=False)
with open('results_best_params.json', 'w') as fh:
    json.dump({f'{r.City}|{r.Model}': str(r._4) for r in cv_results.itertuples()}, fh, indent=2)
print(f'\nsaved {len(written)} pipeline files, submission ({len(sub)} rows)')
print(f'total {time.time()-t0:.0f}s')
