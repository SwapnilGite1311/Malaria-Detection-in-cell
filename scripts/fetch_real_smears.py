"""Download a small sample of REAL blood smear photos with expert labels.

Source: BBBC041 P. vivax Malaria Blood Smear Dataset, Broad Institute
https://bbbc.broadinstitute.org/BBBC041  (CC BY-NC-SA 3.0, by Jane Hung)

The full zip is 2.26 GB. A zip file keeps a "table of contents" at its end, and
the server lets us download any byte range, so we read the table of contents and
then fetch only the photos we want (about 1.3 MB each).

Run:  python -m scripts.fetch_real_smears
"""
import json
import random
import urllib.request
import zipfile

from src import config

URL = "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"
OUT_DIR = config.PROJECT_DIR / "real_smears"
N_IMAGES = 40
INFECTED_CATEGORIES = {"ring", "trophozoite", "schizont", "gametocyte"}


class RemoteFile:
    """Looks like an open file to zipfile, but each read() downloads just those bytes."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        self.downloaded = 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        start = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = start + offset
        return self.position

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.position
        if n == 0 or self.position >= self.size:
            return b""
        last_byte = min(self.position + n, self.size) - 1
        request = urllib.request.Request(self.url,
                                         headers={"Range": f"bytes={self.position}-{last_byte}"})
        data = urllib.request.urlopen(request, timeout=120).read()
        self.position += len(data)
        self.downloaded += len(data)
        return data


def parasitemia(entry):
    """The experts' answer: infected red cells / all red cells (white cells excluded)."""
    categories = [obj["category"] for obj in entry["objects"]]
    infected = sum(c in INFECTED_CATEGORIES for c in categories)
    red_cells = infected + categories.count("red blood cell")
    return infected / red_cells if red_cells else 0.0


def main():
    remote = RemoteFile(URL)
    archive = zipfile.ZipFile(remote)

    # The dataset's own test split: photos the dataset authors kept aside for testing
    entries = json.loads(archive.read("malaria/test.json"))

    # Pick photos across the whole range of infection levels, not just easy ones
    entries.sort(key=parasitemia)
    step = len(entries) / N_IMAGES
    chosen = [entries[int(i * step)] for i in range(N_IMAGES)]
    random.Random(config.SEED).shuffle(chosen)

    image_dir = OUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for number, entry in enumerate(chosen, start=1):
        name = entry["image"]["pathname"].lstrip("/")          # "images/abc.png"
        target = OUT_DIR / name
        if not target.exists():
            target.write_bytes(archive.read(f"malaria/{name}"))
        print(f"[{number}/{N_IMAGES}] {target.name}  parasitemia {parasitemia(entry):.1%}",
              flush=True)

    with open(OUT_DIR / "labels.json", "w") as f:
        json.dump(chosen, f, indent=1)
    print(f"Done: {remote.downloaded / 1e6:.1f} MB downloaded into {OUT_DIR}")


if __name__ == "__main__":
    main()
