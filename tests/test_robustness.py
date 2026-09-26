import csv
import json
import tempfile
import unittest
from pathlib import Path

from ber.common import FIELDS, TSV, Record, normalize, partition_anchors, records, audit_inputs
from ber.views import address_parts, core_name, views
from ber.retrieval import build_index, Retriever, RetrievalConfig
from ber.language import mine_language
from ber.demo import make_demo
from ber.training import prepare_pairs, train, evaluate_trust
from ber.inference import predict, validate, fallback
from ber.features import features, FEATURE_NAMES
from ber.decision import utility_set


def write_records(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, **TSV, lineterminator="\n")
        w.writerow(FIELDS); w.writerows(rows)


class RobustnessTests(unittest.TestCase):
    def test_literal_quotes_nulls_digits_and_numbers(self):
        self.assertEqual(normalize('२० Main Rd NULL N/A nan none'), '20 main rd')
        self.assertEqual(normalize('Nullarbor Annan'), 'nullarbor annan')
        self.assertEqual(views('X', '8 St Michel', 'France')['expanded'], '8 saint michel')
        self.assertEqual(views('X', '8 St Michel', 'US')['expanded'], '8 street michel')
        self.assertEqual(core_name('Baba प्राइवेट लिमिटेड', 'India'), 'baba')
        self.assertEqual(core_name("L’Étoile Société SAS", 'France'), 'etoile')
        self.assertIn('example', views('Vendor [www.example.com]', '', 'India')['aliases'])
        self.assertEqual(address_parts('20085-20089 Main Rd, Suite 5', 'US')['house_hi'], 20089)
        self.assertEqual(address_parts('२० bis rue Neuve', 'France')['house_suffix'], 'bis')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'source.tsv'
            write_records(p, [['S1-1', '"Quoted name', '1 Road', 'US'], ['S1-2', 'Other', 'NULL', 'US']])
            rows = list(records(p))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].business_name, '"Quoted name')
            self.assertEqual(rows[1].a, '')

    def test_ranked_common_bucket_and_null_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source.tsv'
            rows = [[f'S2-{i}', 'Common Business', f'{9000+i} Wrong Road', 'US'] for i in range(100)]
            rows += [['S2-best', 'Common Business', '20 Right Road', 'US'], ['S2-null', 'Other Name', 'NULL', 'US']]
            write_records(source, rows)
            build_index([source], root/'index')
            r = Retriever(root/'index', RetrievalConfig(per_channel=2, per_source=8, max_per_source=16))
            cs = r.retrieve(Record('S1-1', 'Common Business', '20 Right Road', 'US'))
            self.assertIn('S2-best', [c.record.entity_id for c in cs])
            r.retrieve(Record('S1-2', 'Unrelated', 'NULL', 'US'))
            self.assertEqual(r.last_channels['exact_a'], set())
            with self.assertRaisesRegex(ValueError, 'Unknown country'):
                r.retrieve(Record('S1-3', 'Common', '', 'Unknown'))
            self.assertEqual(len(r.exact_union(Record('S1-4', 'Common Business', '', 'US'))), 101)
            r.close()

    def test_language_split_guard_parallel_resume_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = make_demo(root/'data', train_per_country=160, test_per_country=4)
            audit_inputs(data)
            language = root/'language.json'
            artifact = mine_language(data/'train/train_source1.tsv', data/'train/train_ground_truth.tsv',
                [data/'train'/f'train_source{s}.tsv' for s in (2, 3)], language, per_country=160)
            self.assertEqual(artifact['provenance'], 'training fold only; no calibration/trust labels')
            for split in ('train', 'test'):
                build_index([data/split/f'{split}_source{s}.tsv' for s in (2, 3)], root/split, language)
            run = root/'run'
            prepare_pairs(data/'train/train_source1.tsv', data/'train/train_ground_truth.tsv', root/'train', run,
                          per_country=160, max_pairs=100000, retrieval=RetrievalConfig(per_channel=5, per_source=12, max_per_source=20))
            with self.assertRaisesRegex(ValueError, 'WITHOUT held-country'):
                train(run, max_iter=15, exclude_country='India')
            model, _ = train(run, max_iter=25)
            evaluate_trust(run, model)
            serial, parallel = root/'serial', root/'parallel'
            predict(data/'test/test_source1.tsv', root/'test', model, serial, batch_size=4)
            predict(data/'test/test_source1.tsv', root/'test', model, parallel, batch_size=4, workers=2)
            self.assertEqual((serial/'matching_results.tsv').read_bytes(), (parallel/'matching_results.tsv').read_bytes())
            predict(data/'test/test_source1.tsv', root/'test', model, parallel, batch_size=4, workers=2)
            self.assertEqual(validate(data/'test/test_source1.tsv', root/'test', parallel)['status'], 'PASS')
            fallback(data/'test/test_source1.tsv', root/'test', root/'fallback')
            self.assertEqual(validate(data/'test/test_source1.tsv', root/'test', root/'fallback')['status'], 'PASS')

    def test_duplicate_group_not_split(self):
        anchors = [Record(f'S1-{i}', f'Name {i}', '1 Rd', 'US') for i in range(100)]
        anchors.append(Record('S1-copy', 'Name 3', '1 Rd', 'US'))
        truth = {a.entity_id: {'S2-1'} if i%2 else set() for i, a in enumerate(anchors)}
        split = partition_anchors(anchors, truth)
        self.assertEqual(split['S1-copy'], split['S1-3'])
        self.assertEqual(set(split.values()), {'train', 'calibration', 'trust'})

    def test_case_review_patterns_and_perturbations(self):
        # Synthetic regression cases inspired by the six supplied audit examples.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'targets.tsv'
            rows = [
                ['S2-baba', 'बाबा इंफोटेक प्राइवेट लिमिटेड', '', 'India'],
                ['S2-alpha', 'Limited Constructions Alpha Private', '21 Kolkata Road', 'India'],
                ['S2-alias', 'Arcyuma', '8 Evansville Road', 'US'],
                ['S2-compact', '#DesrochersPiedmontHillofRockford', '', 'US'],
                ['S2-volt', 'Other Brand', '9 Mumbai Road', 'India'],
                ['S2-ap', 'AP Hospitality', '20085-20089 Main Street', 'US'],
                ['S2-fr', 'Etoile SARL', '20 bis Rue Neuve 75008 Paris', 'France'],
            ]
            write_records(source, rows)
            language = root/'language.json'
            language.write_text(json.dumps({'aliases': {'बाबा इंफोटेक प्राइवेट लिमिटेड': ['baba infotech private limited']}, 'characters': {}}))
            build_index([source], root/'index', language)
            r = Retriever(root/'index')
            anchors = [
                ('Baba Infotech Private Limited', '', 'India', 'S2-baba'),
                ('Alpha Constructions Private Limited', '21 Kolkata Rd', 'India', 'S2-alpha'),
                ('Donahue Solar LLC', '8 Evansville Road', 'US', 'S2-alias'),
                ('Desrochers Piedmont Hill of Rockford Inc', '', 'US', 'S2-compact'),
                ('Volt Welfare Society', '9 Mumbai Road', 'India', 'S2-volt'),
                ('AP Hospitality Inc', '20085 Main St', 'US', 'S2-ap'),
                ('Étoile Société SAS', '20 bis R. Neuve 75008 Paris CEDEX', 'France', 'S2-fr'),
            ]
            for name, address, country, expected in anchors:
                a = Record('S1-case', name, address, country)
                cs = r.retrieve(a)
                self.assertIn(expected, [c.record.entity_id for c in cs])
                matrix = features(a, cs)
                if expected == 'S2-ap':
                    self.assertEqual(matrix[0, FEATURE_NAMES.index('house_range_overlap')], 1)
                if expected == 'S2-volt':
                    self.assertLess(matrix[0, FEATURE_NAMES.index('name_ratio')], .5)
            r.close()
        self.assertEqual(utility_set(['a'], [.99], 0, .99), ['a'])
        self.assertEqual(utility_set(['a'], [.01], 0, .01), [])


if __name__ == '__main__': unittest.main()
