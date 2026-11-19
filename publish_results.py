"""Publish RESULTS.md + results.json to the GitHub repo from Kaggle.

Requires a GitHub Personal Access Token stored as a Kaggle Secret named GH_TOKEN
(Notebook settings -> Secrets -> Add secret). The token is read from the
environment at runtime — it is never written into any file in the repo.
"""
import json
import os
import shutil
import subprocess
import sys

REPO_DIR = "DIT5411-LuiYuHin"
REPO_URL = "https://github.com/cveg1234/DIT5411-LuiYuHin.git"
TOKEN = os.environ.get("GH_TOKEN")


def main():
    if not TOKEN:
        print("ERROR: GH_TOKEN not found. In Kaggle: Settings -> Secrets -> "
              "Add secret named GH_TOKEN with a GitHub personal access token "
              "(repo scope). Then re-run this cell.")
        return

    results = os.path.join(REPO_DIR, "results")
    if not os.path.isfile(os.path.join(results, "RESULTS.md")):
        print("ERROR: results/RESULTS.md not found — finish training first.")
        return

    # Copy generated results into the repo working tree
    shutil.copy2(os.path.join(results, "RESULTS.md"),
                 os.path.join(REPO_DIR, "results", "RESULTS.md"))
    shutil.copy2(os.path.join(results, "results.json"),
                 os.path.join(REPO_DIR, "results", "results.json"))
    shutil.copy2(os.path.join(results, "label_map.json"),
                 os.path.join(REPO_DIR, "results", "label_map.json"))

    # Update README Results section with the accuracy table
    readme_path = os.path.join(REPO_DIR, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()
    with open(os.path.join(results, "RESULTS.md"), encoding="utf-8") as f:
        table = f.read()
    marker_start = "## 4. Results"
    marker_end = "---\n\n## 5. How to run"
    i, j = readme.find(marker_start), readme.find(marker_end)
    if i != -1 and j != -1:
        new_section = (
            "## 4. Results\n\n"
            + table.strip()
            + "\n\n"
            + "> Model weights and checkpoints are kept in the Kaggle notebook "
              "Output, not in this repo (GitHub's 100 MB file limit).\n\n"
        )
        readme = readme[:i] + new_section + readme[j:]
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)

    def git(*args):
        r = subprocess.run(["git", "-C", REPO_DIR, *args],
                           capture_output=True, text=True)
        print(f"$ git {' '.join(args)}")
        if r.stdout:
            print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr.strip())
        return r.returncode

    # Authenticate with token via URL (never printed)
    auth_url = REPO_URL.replace("https://", f"https://{TOKEN}@")
    git("remote", "set-url", "origin", auth_url)
    git("add", "README.md", "results/RESULTS.md",
        "results/results.json", "results/label_map.json")
    git("commit", "-m", "Add model comparison results and accuracy table")
    rc = git("push", "origin", "main")
    git("remote", "set-url", "origin", REPO_URL)  # remove token from config
    print("Push", "SUCCEEDED" if rc == 0 else "FAILED")
    print("Repo: https://github.com/cveg1234/DIT5411-LuiYuHin")


if __name__ == "__main__":
    main()
