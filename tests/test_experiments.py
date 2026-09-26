import json
import tempfile
import unittest
from pathlib import Path

from ber.demo import make_demo
from ber.retrieval import build_index, RetrievalConfig
from ber.training import prepare_pairs, train, load_model, load_part, predict_proba
from ber.experiments import model_sweep, paired_comparison, soft_ownership, register_run


class ExperimentTests(unittest.TestCase):
    def test_optional_model_paths_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = make_demo(root/'data', train_per_country=180, test_per_country=2)
            build_index([data/'train'/f'train_source{s}.tsv' for s in (2, 3)], root/'index')
            run = root/'run'
            prepare_pairs(data/'train/train_source1.tsv', data/'train/train_ground_truth.tsv', root/'index', run,
                retrieval=RetrievalConfig(per_channel=5, per_source=10, max_per_source=16), per_country=180, max_pairs=100000)
            configs = [dict(max_iter=12, save_oof=True, rerank=True),
                       dict(max_iter=12, calibration='isotonic', early_stopping=True),
                       dict(max_iter=12, objective='ranking'), dict(max_iter=12, bag_seeds=[2027])]
            result = model_sweep(run, configs, root/'variants')
            self.assertEqual(len(result), 4)
            for entry in result:
                bundle = load_model(Path(entry['path'])/'model.pkl')
                x, _, _, rows = load_part(entry['path'], 'calibration')
                probs = predict_proba(bundle, x, rows)
                self.assertTrue(((probs >= 0) & (probs <= 1)).all())
                self.assertFalse((Path(entry['path'])/'trust_metrics.json').exists())
            register_run(result[0]['path'], root/'registry', leaderboard=.7, submission='synthetic-check')
            self.assertTrue((root/'registry/comparison.tsv').exists())
            rows = [{'group':'a','candidates':['S2-1'],'probabilities':[.9]},
                    {'group':'b','candidates':['S2-1'],'probabilities':[.8]}]
            updated = soft_ownership(rows)
            self.assertLess(updated[0]['probabilities'][0], .9)
            self.assertEqual(updated[0]['candidates'], ['S2-1'])


if __name__ == '__main__': unittest.main()
