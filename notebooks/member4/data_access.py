"""Development-only Member 4 inputs. Never fit preprocessing in this module."""
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DATASETS = {'shanghai': (80, 20, 25), 'cgmacros': (36, 9, 39)}
FORBIDDEN = {'participant_id', 'timestamp', 'target_timestamp_30min', 'target_glucose_30min'}
TARGET = 'target_glucose_30min'

def require(condition, message):
    if not condition:
        raise ValueError(message)

@dataclass
class DevelopmentData:
    dataset: str
    frame: pd.DataFrame
    feature_metadata: pd.DataFrame
    assignments: pd.DataFrame
    development_ids: frozenset
    locked_ids: frozenset
    excluded_rows: int

    @property
    def features(self):
        return self.feature_metadata['feature'].tolist()

    def xy(self, frame=None):
        frame = self.frame if frame is None else frame
        require(frame['participant_id'].isin(self.development_ids).all(), 'Non-development participant in X/y request')
        require(not set(self.features) & FORBIDDEN, 'Forbidden predictive feature')
        X = frame.loc[:, self.features].copy()
        require(list(X.columns) == self.features, 'Incorrect feature order')
        return X, frame[TARGET].copy()

    def fold(self, number):
        require(number in range(1, 6), 'Fold must be 1 through 5')
        valid = frozenset(self.assignments.loc[self.assignments.validation_fold == number, 'participant_id'])
        train = self.development_ids - valid
        require(bool(train) and bool(valid) and train.isdisjoint(valid), 'Invalid fold separation')
        require((train | valid).isdisjoint(self.locked_ids), 'Locked participant in fold')
        a = self.frame.loc[self.frame.participant_id.isin(train)].copy()
        b = self.frame.loc[self.frame.participant_id.isin(valid)].copy()
        return a, b

def load_development(source_root, dataset, chunksize=50000):
    require(dataset in DATASETS, 'Unknown dataset')
    root = Path(source_root).resolve()
    ndev, ntest, nfeatures = DATASETS[dataset]
    split = pd.read_csv(root / f'reports/modeling/{dataset}_split.csv', dtype={'participant_id': 'string'})
    folds = pd.read_csv(root / f'reports/modeling/{dataset}_development_folds.csv', dtype={'participant_id': 'string'})
    meta = pd.read_csv(root / f'reports/data_audit/{dataset}_final_feature_list.csv').sort_values('feature_number').reset_index(drop=True)
    for table in (split, folds):
        require(table.participant_id.notna().all() and table.participant_id.str.strip().ne('').all(), 'Missing participant ID in assignments')
        require(table.participant_id.is_unique, 'Duplicate participant assignment')
    require(set(split.dataset_split) == {'development', 'test'}, 'Unexpected split labels')
    dev = frozenset(split.loc[split.dataset_split == 'development', 'participant_id'])
    test = frozenset(split.loc[split.dataset_split == 'test', 'participant_id'])
    require(len(dev) == ndev and len(test) == ntest and dev.isdisjoint(test), 'Invalid participant split')
    require(set(folds.participant_id) == dev and len(folds) == ndev, 'Fold IDs must equal development IDs exactly')
    require(set(folds.validation_fold) == set(range(1, 6)), 'Invalid validation folds')
    require(len(meta) == nfeatures and meta.feature.is_unique, 'Invalid approved feature count/names')
    require(meta.feature_number.tolist() == list(range(1, nfeatures + 1)), 'Invalid feature order')
    require(meta.model_role.eq('input_X').all() and meta.uses_future_information.eq('No').all(), 'Unapproved feature role/future information')
    features = meta.feature.tolist()
    require(not FORBIDDEN & set(features), 'Metadata or target in approved X')
    columns = ['participant_id', 'timestamp', 'target_timestamp_30min'] + features + [TARGET]
    chunks, excluded = [], 0
    with pd.read_csv(root / f'data/processed/{dataset}_ml_ready.csv', usecols=columns,
                     dtype={'participant_id': 'string'}, chunksize=chunksize) as reader:
        for chunk in reader:
            # Fail on unmapped IDs; never quietly classify them as locked test rows.
            require(chunk.participant_id.notna().all() and chunk.participant_id.isin(dev | test).all(), 'Unmapped/missing participant ID in source')
            mask = chunk.participant_id.isin(dev)
            excluded += int((~mask).sum())
            kept = chunk.loc[mask, columns].copy()
            if len(kept):
                chunks.append(kept)
    require(bool(chunks), 'No development rows')
    frame = pd.concat(chunks, ignore_index=True)
    require(set(frame.participant_id) == dev, 'Development participants missing from source')
    result = DevelopmentData(dataset, frame, meta, folds, dev, test, excluded)
    result.xy()
    for number in range(1, 6):
        a, b = result.fold(number)
        require(set(a.participant_id).isdisjoint(set(b.participant_id)), 'Row partition overlap')
        require(len(a) + len(b) == len(frame), 'Incomplete fold partition')
    return result
