"""Read source artifacts; write only Member 4 audit outputs. No fit/predict calls."""
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT.parent / 'Smart_Glucose_Work'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_csv(name, records):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

protected = sorted(p for folder in ['reports', 'results', 'models', 'notebooks', 'data/processed']
                   for p in (SOURCE / folder).rglob('*') if p.is_file())
baseline = {p: digest(p) for p in protected}
save_csv('member4_protected_checksums.csv', [dict(source_relative_path=p.relative_to(SOURCE).as_posix(),
         size_bytes=p.stat().st_size, sha256=h) for p, h in baseline.items()])

checks = []
for manifest, key in [('reports/modeling/asly_artifact_manifest.csv', 'relative_path'),
                      ('reports/evaluation/member3_integrity_manifest.csv', 'file_path'),
                      ('reports/evaluation/member3_trained_model_inventory.csv', 'model_file')]:
    for r in rows(SOURCE / manifest):
        p = SOURCE / r[key]
        actual = baseline.get(p) or (digest(p) if p.is_file() else '')
        checks.append(dict(manifest=manifest, artifact=r[key], exists=p.is_file(),
                           expected_sha256=r['sha256'], actual_sha256=actual,
                           matches=actual.lower() == r['sha256'].lower()))
save_csv('member4_manifest_verification.csv', checks)

summary = []
for dataset, ndev, ntest, nf in [('shanghai',80,20,25), ('cgmacros',36,9,39)]:
    split = rows(SOURCE / f'reports/modeling/{dataset}_split.csv')
    folds = rows(SOURCE / f'reports/modeling/{dataset}_development_folds.csv')
    features = sorted(rows(SOURCE / f'reports/data_audit/{dataset}_final_feature_list.csv'), key=lambda r:int(r['feature_number']))
    dev = {r['participant_id'] for r in split if r['dataset_split']=='development'}
    test = {r['participant_id'] for r in split if r['dataset_split']=='test'}
    assert all(r['participant_id'] for r in split + folds)
    assert len(split)==len({r['participant_id'] for r in split})==ndev+ntest
    assert {r['dataset_split'] for r in split}=={'development','test'}
    assert len(dev)==ndev and len(test)==ntest and dev.isdisjoint(test)
    assert len(folds)==len({r['participant_id'] for r in folds})==ndev
    assert {r['participant_id'] for r in folds}==dev
    assert {r['validation_fold'] for r in folds}==set('12345')
    assert len(features)==len({r['feature'] for r in features})==nf
    assert [int(r['feature_number']) for r in features]==list(range(1,nf+1))
    assert all(r['uses_future_information']=='No' and r['model_role']=='input_X' for r in features)
    assert not {'participant_id','timestamp','target_timestamp_30min','target_glucose_30min'} & {r['feature'] for r in features}
    sizes=Counter(r['validation_fold'] for r in folds)
    comparison=rows(SOURCE / f'results/evaluation/{dataset}_ridge_feature_group_comparison.csv')
    assert len(comparison)==(5 if dataset=='shanghai' else 6)
    assert all(r['evaluation_scope']=='development_only' for r in comparison)
    # Header only: no processed participant rows are inspected.
    with (SOURCE / f'data/processed/{dataset}_ml_ready.csv').open(encoding='utf-8-sig',newline='') as f:
        header=next(csv.reader(f))
    assert set(r['feature'] for r in features).issubset(header)
    metric_files=list((SOURCE/'results/evaluation').glob(f'{dataset}_ridge_*_fold[1-5].csv'))
    for p in metric_files:
        rr=rows(p)
        assert len(rr)==1 and rr[0]['evaluation_scope']=='development_only'
        r=rr[0]; nv=sizes[r['validation_fold']]
        assert int(r['validation_participants'])==nv and int(r['train_participants'])==ndev-nv
    summary.append(dict(dataset=dataset,development=ndev,locked_test=ntest,features=nf,
                        validation_sizes=dict(sorted(sizes.items())),metric_files=len(metric_files),
                        lowest_mae=min(comparison,key=lambda r:float(r['MAE_mean']))['experiment']))

environment=[]
for package,module in [('pandas','pandas'),('numpy','numpy'),('scikit-learn','sklearn'),('joblib','joblib'),
                       ('matplotlib','matplotlib'),('scipy','scipy'),('shap','shap'),('seaborn','seaborn'),('nbformat','nbformat'),('jupyter','jupyter')]:
    try:
        version=importlib.metadata.version(package)
        importlib.import_module(module)
        status='import OK'
    except Exception as e:
        version=locals().get('version','')
        status=f'{type(e).__name__}: {e}'
    environment.append(dict(package=package,version=version,status=status))
    version=''
save_csv('member4_environment_check.csv',environment)

models=[]
try:
    import joblib
    for p in sorted((SOURCE/'models').glob('*.joblib')):
        model=joblib.load(p)
        dataset=p.name.split('_')[0]
        expected=[r['feature'] for r in sorted(rows(SOURCE/f'reports/data_audit/{dataset}_final_feature_list.csv'),key=lambda r:int(r['feature_number']))]
        assert list(model.feature_names_in_)==expected
        pre=model.named_steps['preprocessor']; est=model.named_steps['model']
        names=list(pre.get_feature_names_out())
        models.append(dict(file=p.name,estimator=type(est).__name__,input_features=len(expected),
                           transformed_features=len(names),alpha=getattr(est,'alpha',None),
                           preprocessing=repr(pre)))
except Exception as e:
    models.append(dict(error=f'{type(e).__name__}: {e}'))
(OUT/'member4_model_inspection.json').write_text(json.dumps(models,indent=2),encoding='utf-8')

nb=json.loads((SOURCE/'notebooks/04_asly_model_development.ipynb').read_text(encoding='utf-8'))
evidence=[]
for i,cell in enumerate(nb['cells']):
    text=''.join(cell.get('source',[]))
    if any(t in text for t in ['joblib.dump','best_estimator_','cg_svr_final_pipeline.fit']):
        evidence.append(f'Cell {i} (zero-based):\n{text}')
(OUT/'member4_model_provenance_evidence.txt').write_text('\n\n'.join(evidence),encoding='utf-8')

changed=[p.relative_to(SOURCE).as_posix() for p,h in baseline.items() if not p.exists() or digest(p)!=h]
assert not changed, changed
payload=dict(date=datetime.now(ZoneInfo('Asia/Colombo')).isoformat(),source=str(SOURCE),
             python=platform.python_version(),protected_files=len(baseline),
             manifest_checks=len(checks),manifest_mismatches=sum(not r['matches'] for r in checks),
             datasets=summary,model_files=list(p.name for p in (SOURCE/'models').rglob('*.joblib')),
             protected_changed=changed)
(OUT/'member4_readiness_evidence.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
print(json.dumps(payload,indent=2))
