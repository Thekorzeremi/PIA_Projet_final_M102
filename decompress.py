from pathlib import Path
import gzip
import shutil

def decompress_gz(fichier_entree: Path):
    with gzip.open(fichier_entree, 'rb') as f_in:
        with open(fichier_entree.with_suffix(''), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    path = Path("./listings.csv.gz")
    decompress_gz(path)