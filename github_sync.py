"""Sync results between a Kaggle notebook and GitHub.

    python github_sync.py pull   # adopt results already published on GitHub
    python github_sync.py push   # publish local results (needs GH_TOKEN)

pull uses raw.githubusercontent.com (public repo, no token). push reads a GitHub
token from the GH_TOKEN environment variable (Kaggle Secret) — never written
to any file. Both are no-ops when content is unchanged.
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request

REPO_DIR = "DIT5411-LuiYuHin"
REPO_URL = "https://github.com/cveg1234/DIT5411-LuiYuHin.git"
RAW = "https://raw.githubusercontent.com/cveg1234/DIT5411-LuiYuHin/main/results/"
FILES = ["results.json", "RESULTS.md", "label_map.json"]


def pull():
    os.makedirs(os.path.join(REPO_DIR, "results"), exist_ok=True)
    for name in FILES:
        dst_local = os.path.join("results", name)
        dst_repo = os.path.join(REPO_DIR, "results", name)
        try:
            with urllib.request.urlopen(RAW + name, timeout=20) as r:
                data = r.read()
        except Exception as e:
            print(f"  {name}: not on GitHub yet ({e.__class__.__name__})")
            continue
        changed = False
        for dst in (dst_local, dst_repo):
            if not os.path.exists(dst) or open(dst, "rb").read() != data:
                with open(dst, "wb") as f:
                    f.write(data)
                changed = True
        print(f"  {name}: {'adopted from GitHub' if changed else 'already up to date'}")
    rp = os.path.join("results", "results.json")
    if os.path.isfile(rp):
        data = json.load(open(rp, encoding="utf-8"))
        print("  Published models:", [m["model"] for m in data.get("models", [])])


def push():
    token = os.environ.get("GH_TOKEN")
    if not token:
        print("ERROR: GH_TOKEN missing. Add it in Kaggle: Settings -> Secrets.")
        return 1
    src, dst = "results", os.path.join(REPO_DIR, "results")
    if not os.path.isfile(os.path.join(src, "RESULTS.md")):
        print("ERROR: results/RESULTS.md not found — finish a model first.")
        return 1

    changed = []
    for name in FILES:
        s = os.path.join(src, name)
        if not os.path.isfile(s):
            continue
        d = os.path.join(dst, name)
        if not os.path.exists(d) or open(s, "rb").read() != open(d, "rb").read():
            shutil.copy2(s, d)
            changed.append(name)
    print("Changed files:", changed if changed else "none")

    if "RESULTS.md" in changed:
        readme_path = os.path.join(REPO_DIR, "README.md")
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()
        table = open(os.path.join(dst, "RESULTS.md"), encoding="utf-8").read().strip()
        start, end = "## 4. Results", "---\n\n## 5. How to run"
        i, j = readme.find(start), readme.find(end)
        if i != -1 and j != -1:
            section = ("## 4. Results\n\n" + table + "\n\n"
                       "> Model weights and checkpoints remain in the Kaggle "
                       "notebook Output (GitHub's 100 MB file limit).\n\n")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme[:i] + section + readme[j:])
            changed.append("README.md")
            print("Updated README accuracy section")

    if not changed:
        print("Nothing to push — GitHub already has these results.")
        return 0

    def git(*args):
        r = subprocess.run(["git", "-C", REPO_DIR, *args],
                           capture_output=True, text=True)
        print(f"$ git {' '.join(args)}")
        if r.stdout:
            print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr.strip())
        return r.returncode

    auth_url = REPO_URL.replace("https://", f"https://{token}@")
    git("remote", "set-url", "origin", auth_url)
    paths = ["results/" + n for n in FILES
             if os.path.isfile(os.path.join(dst, n))]
    git("add", *paths)
    if "README.md" in changed:
        git("add", "README.md")
    git("commit", "-m", "Update model comparison results")
    rc = git("push", "origin", "main")
    git("remote", "set-url", "origin", REPO_URL)
    print("Push", "SUCCEEDED" if rc == 0 else "FAILED")
    print("Repo: https://github.com/cveg1234/DIT5411-LuiYuHin")
    return rc


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "push"
    if mode == "pull":
        pull()
    else:
        sys.exit(push())
