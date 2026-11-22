# Publishing to GitHub

The project folder is `DIT5411-LuiYuHin`. The repository must be named
`DIT5411-LuiYuHin` (assignment: "DIT5411-your name", your name = LuiYuHin).

## 1. Initialise locally

```bash
cd DIT5411-LuiYuHin
git init
git add README.md config.py dataset.py augmentation.py generators.py models.py train.py predict.py requirements.txt .gitignore notebooks/
git commit -m "Initial DIT5411 character recognition project"
```

`data/` and `results/*.keras` are gitignored, so the dataset and trained model
weights will not be pushed. `results/RESULTS.md` and `results/results.json`
are small and should be committed after training.

## 2. Create the remote repository

On GitHub: **New repository** → name `DIT5411-LuiYuHin` → Public →
**do not** initialise with a README (one already exists).

Then:

```bash
git remote add origin https://github.com/cveg1234/DIT5411-LuiYuHin.git
git branch -M main
git push -u origin main
```

## 3. After the full training run

```bash
git add results/RESULTS.md results/results.json
git commit -m "Add model comparison results"
git push
```

## Checklist for the submitted repo

- [ ] README with dataset, augmentation method, model descriptions, references
- [ ] Source code files listed above
- [ ] `results/RESULTS.md` with the measured accuracy table
- [ ] `results/results.json` with full metrics
- [ ] Kaggle training notebook in `notebooks/`
- [ ] Repo name `DIT5411-LuiYuHin`
