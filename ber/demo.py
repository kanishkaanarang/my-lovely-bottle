"""Generated fixture only. Never use demo metrics as challenge performance."""

import csv
from pathlib import Path

from .common import FIELDS, TSV


def make_demo(root, train_per_country=160, test_per_country=12):
    root = Path(root)
    for split, countries, count in (("train", ("US", "India"), train_per_country),
                                     ("test", ("US", "India", "France"), test_per_country)):
        folder = root/split
        folder.mkdir(parents=True, exist_ok=True)
        data = {1: [], 2: [], 3: []}
        truth = []
        target_counter = 1 if split == "train" else 1_000_000
        for country_no, country in enumerate(countries):
            for i in range(count):
                number = (0 if split == "train" else 1_000_000) + country_no*100_000+i
                aid = f"S1-{number:09d}"
                brand = ["Harbor", "Maple", "Cedar", "River", "Summit"][i % 5]
                tail = chr(97+(i//26)%26) + chr(97+i%26)
                name = f"{brand} {tail} Services Pvt Ltd" if country == "India" else f"{brand} {tail} Labs Inc"
                if country == "France":
                    name = f"Étoile {tail} Services SARL"
                address = f"{1000+i*7} Orchard Road, District {i%13}, {country}"
                data[1].append([aid, name, address, country])
                ids = []
                if i % 9:
                    for src, variation in ((2, 0), (2, 1), (3, 2)):
                        target_counter += 1
                        tid = f"S{src}-{target_counter:09d}"
                        n = name.replace("Inc", "Incorporated").replace("Pvt", "Private")
                        a = address.replace("Road", "Rd").upper() if src == 2 else address
                        if variation == 1:
                            n = " ".join(reversed(n.split()))
                            if i % 11 == 0:
                                a = ""
                        if variation == 2 and country == "India" and i % 7 == 0:
                            n = "नदी सेवा " + tail
                        if country == "France":
                            n = n.replace("É", "E")
                        data[src].append([tid, n, a, country]); ids.append(tid)
                # Exact-name hard negative, elsewhere. Present for singletons too.
                target_counter += 1
                data[2].append([f"S2-{target_counter:09d}", name, f"{90000+i} Industrial Lane, Other Town", country])
                truth.append([aid, ",".join(ids)])
        for src, rows in data.items():
            with (folder/f"{split}_source{src}.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, **TSV, lineterminator="\n")
                writer.writerow(FIELDS); writer.writerows(rows)
        if split == "train":
            with (folder/"train_ground_truth.tsv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, **TSV, lineterminator="\n")
                writer.writerow(["source1_entity_id", "matched_entity_ids"]); writer.writerows(truth)
    return root
