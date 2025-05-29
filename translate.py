from pathlib import Path

translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}

for p in Path("images").iterdir():
    if p.is_dir() and translate.get(p.name):
        p.rename("images/" + translate[p.name])
