from pathlib import Path
#from database import CasinoDatabase
from config.paths import Paths
import sqlite3

import random
import datetime
import math

def afficher_structure(path: Path, prefix=""):
    contenu = sorted(path.iterdir())
    for i, item in enumerate(contenu):
        is_last = i == len(contenu) - 1
        branche = "└── " if is_last else "├── "
        print(prefix + branche + item.name)
        if item.is_dir():
            extension = "    " if is_last else "│   "
            afficher_structure(item, prefix + extension)

afficher_structure(Path("."))
