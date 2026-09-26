import csv
import json
import math
import re
import shutil
import tempfile
import unittest
import zipfile
import subprocess
import os
import sys
from pathlib import Path

from ber.aws import extract_resource
from ber.common import Record, entity_f05, file_hash, fold, fold_latin, normalize, records
from ber.demo import make_demo
from ber.inference import package_submission, predict, validate
from ber.retrieval import Retriever, RetrievalConfig, build_index
from ber.training import evaluate_trust, prepare_pairs, train


class CoreTests(unittest.TestCase):
    def test_metric_and_unicode(self):
        for t, p, expected in [([], [], 1), ([], ['x'], 0), (['a'], [], 0),
                               (['a', 'b'], ['a'], 5/6), (['a', 'b'], ['a', 'b', 'c'], 5/7)]:
            self.assertTrue(math.isclose(entity_f05(t, p), expected))
        self.assertEqual(normalize('बाबा इंफोटेक'), 'बाबा इंफोटेक')
        self.assertEqual(fold_latin('Étoile बाबा'), 'Etoile बाबा')
        a = Record('S1-1', 'One Inc.', '9 Road', 'New country')
        b = Record('S1-2', 'ONE INC', '9 ROAD', 'New country')
        self.assertEqual(fold(a), fold(b))

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('student_resource/../../outside.txt', 'bad')
            with self.assertRaises(ValueError):
                extract_resource(path, Path(tmp)/'extract')


class EndToEndTests(unittest.TestCase):
    def test_train_predict_resume_validate_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_demo(root/'data', train_per_country=160, test_per_country=8)
            train_index, test_index = root/'index_train', root/'index_test'
            for split, folder in [('train', train_index), ('test', test_index)]:
                result = build_index([data/split/f'{split}_source{s}.tsv' for s in (2, 3)], folder)
                self.assertEqual(result, build_index([data/split/f'{split}_source{s}.tsv' for s in (2, 3)], folder))
            retrieval = RetrievalConfig(per_channel=8, per_source=16, max_per_source=24)
            run = root/'run'
            prepare_pairs(data/'train/train_source1.tsv', data/'train/train_ground_truth.tsv', train_index, run,
                          retrieval=retrieval, per_country=160, max_pairs=100_000)
            model, _ = train(run, max_iter=35)
            metrics = evaluate_trust(run, model)
            self.assertGreater(metrics['link_recall_ceiling'], .7)
            self.assertGreater(metrics['macro_f05'], .5)
            output = root/'output'
            predict(data/'test/test_source1.tsv', test_index, model, output, batch_size=7)
            before = file_hash(output/'matching_results.tsv')
            predict(data/'test/test_source1.tsv', test_index, model, output, batch_size=7)
            self.assertEqual(before, file_hash(output/'matching_results.tsv'))
            checked = validate(data/'test/test_source1.tsv', test_index, output)
            self.assertEqual(checked['anchors'], 24)  # Includes unseen France.
            repo = Path(__file__).resolve().parents[1]
            package_repo = root/'package_repo'
            package_repo.mkdir()
            shutil.copytree(repo/'ber', package_repo/'ber')
            for name in ('requirements.txt', 'LICENSE', 'SUBMISSION_README.md'):
                shutil.copy2(repo/name, package_repo/name)
            (package_repo/'Documentation_template.md').write_text('Synthetic integration test, not a competition submission.')
            archive = root/'submission.zip'
            package_submission(package_repo, output, model, archive)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('output/candidate_pairs.tsv', z.namelist())
                self.assertIn('code/business_entity_resolution/src/ber/__main__.py', z.namelist())
                self.assertIn('code/business_entity_resolution/model.pkl', z.namelist())
                z.extractall(root/'clean_package')
            packaged = root/'clean_package/code/business_entity_resolution'
            env = dict(os.environ)
            env['PYTHONPATH'] = str(packaged/'src') + os.pathsep + env.get('PYTHONPATH', '')
            clean_index = root/'clean_index'
            rebuilt = subprocess.run([sys.executable, '-m', 'ber', 'index', '--data', str(data),
                '--split', 'test', '--index', str(clean_index), '--language', str(packaged/'language.json')],
                cwd=packaged, env=env, capture_output=True, text=True)
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            clean = subprocess.run([sys.executable, '-m', 'ber', 'predict', '--data', str(data),
                '--index', str(clean_index), '--model', str(packaged/'model.pkl'), '--output', str(root/'clean_output'),
                '--batch-size', '7'], cwd=packaged, env=env, capture_output=True, text=True)
            self.assertEqual(clean.returncode, 0, clean.stderr)
            self.assertEqual((root/'clean_output/matching_results.tsv').read_bytes(), (output/'matching_results.tsv').read_bytes())
            benchmark = root/'benchmark'
            timed = predict(data/'test/test_source1.tsv', test_index, model, benchmark, limit=3)
            self.assertTrue(timed['valid_fresh_benchmark'])
            self.assertEqual(set(timed['drift']), {'US', 'India', 'France'})
            resumed = predict(data/'test/test_source1.tsv', test_index, model, benchmark, limit=3)
            self.assertFalse(resumed['valid_fresh_benchmark'])
            with self.assertRaises(ValueError):
                validate(data/'test/test_source1.tsv', test_index, benchmark)
            lines = (output/'matching_results.tsv').read_text().splitlines()
            lines[1] = lines[1].split('\t')[0]+'\tS2-not-real'
            (output/'matching_results.tsv').write_text('\n'.join(lines)+'\n')
            with self.assertRaises(ValueError):
                validate(data/'test/test_source1.tsv', test_index, output)
            with self.assertRaises(ValueError):
                predict(data/'test/test_source1.tsv', test_index, model, output, batch_size=8)


if __name__ == '__main__':
    unittest.main()
