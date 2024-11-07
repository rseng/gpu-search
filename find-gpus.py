#!/usr/bin/env python

# This is a basic parsing script that will clone repositories
# and search for libraries or mentions that indicate GPUs

from rse.main import Encyclopedia
from rse.utils.command import Command
from rse.utils.file import write_json, read_json, write_file

import matplotlib.pylab as plt
from contextlib import contextmanager
from datetime import datetime
import seaborn as sns
import tempfile
import shutil
import subprocess
import pandas
import argparse
import sys
import os

today = datetime.now()


def clone(url, dest):
    dest = os.path.join(dest, os.path.basename(url))
    cmd = Command("git clone --depth 1 %s %s" % (url, dest))
    cmd.execute()
    if cmd.returncode != 0:
        print("Issue cloning %s" % url)
        return
    return dest


def get_parser():
    parser = argparse.ArgumentParser(
        description="Research Software Encyclopedia Last Updated Analyzer",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--settings-file",
        dest="settings_file",
        help="custom path to settings file.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        help="Output directory for data.",
        default=os.path.join(os.getcwd(), "data"),
    )
    return parser


readme_template = """# %s

```console
%s
```
"""


def find_gpus(repo, dest, outdir):
    result = {"has_gpu": False, "language": repo.data["data"]["language"]}
    with workdir(dest):
        # grep -R -i -E "gpu|cuda|rocm|openacc|opencl"
        p = subprocess.Popen(
            [
                "grep",
                "--exclude-dir=.*",
                "--exclude=*.ipynb",
                "--exclude=*.min.js",
                "--exclude=*.svg",
                "--exclude=*.html",
                "--exclude=*-lock.json",
                "-R",
                "-i",
                "-E",
                "nccl|gpu|cuda|rocm|openacc|opencl|nvidia",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = p.communicate()
        if out is not None:
            out = out.decode("utf-8")
        if out:
            print(f"{repo.url} has GPU evidence")
            result["has_gpu"] = True

            # Save the full data here for inspection
            path = os.path.join(outdir, repo.data["uid"])
            if not os.path.exists(path):
                os.makedirs(path)
            write_file(
                readme_template % (repo.url, out), os.path.join(path, "README.md")
            )
    return result


@contextmanager
def workdir(dirname):
    """
    Provide context for a working directory, e.g.,

    with workdir(name):
       # do stuff
    """
    here = os.getcwd()
    os.chdir(dirname)
    try:
        yield
    finally:
        os.chdir(here)


def main():
    p = get_parser()
    args, extra = p.parse_known_args()

    # Make sure output directory exists, organized by parsing date
    outdir = os.path.abspath(args.outdir)
    if not os.path.exists(outdir):
        print(f"Creating output directory {outdir}")
        os.makedirs(outdir)

    # Create a base temporary folder to work from
    tempdir = tempfile.mkdtemp()

    pedia = Encyclopedia(args.settings_file)
    repos = list(pedia.list())
    total = len(repos)

    # keep track of last updated commit
    meta = {}

    # We will save results as we go, and use cached results for the day if exist
    gpus_json = os.path.join(outdir, "gpus-in-repos.json")

    last_checked = None
    do_check = True
    if os.path.exists(gpus_json):
        meta = read_json(gpus_json)
        last_checked = list(meta.values())[-1]
        do_check = False

    for i, reponame in enumerate(repos):
        print(f"{i} of {total}", end="\r")

        # Prepare to clone the repository
        repo = pedia.get(reponame[0])
        if not repo.url or repo.url in meta:
            print(f"Skipping {repo.url}")
            continue

        if last_checked is not None and not last_checked == repo.url:
            do_check = True

        if not do_check:
            continue

        # Skip javascript languages
        try:
            if repo.data["data"]["language"] in ["Vue", "Javascript"]:
                continue
        except:
            pass

        dest = None
        try:
            # Try clone (and cut out early if not successful)
            dest = clone(repo.url, tempdir)
            if not dest:
                continue
            meta[repo.url] = find_gpus(repo, dest, outdir)

        except:
            print(f"Issue with {repo.url}, skipping")

        try:
            if dest and os.path.exists(dest):
                shutil.rmtree(dest)
        except:
            write_json(meta, gpus_json)
            sys.exit(
                "Likely too many files, check with ulimit -n and set with ulimit -n 4096"
            )

        # Save as we go
        write_json(meta, gpus_json)

    # One final save
    write_json(meta, gpus_json)

    # Do a count across languages - filter out rare languages
    counts = {}
    for repo, data in meta.items():
        language = data["language"]
        if language not in counts:
            counts[language] = 0
        counts[language] += 1

    filtered_group = {k: v for k, v in counts.items() if v > 50}
    filtered_set = set(filtered_group)

    # I meant to filter these - I got the capitalization wrong
    # These have a lot of minimized files that happen to have
    # string matches - we have them in our data but they are
    # erroneous
    filtered_set.remove("JavaScript")
    filtered_set.remove("HTML")

    # Summarize across
    df = pandas.DataFrame(columns=["has_gpu", "language"])

    # Summary will be used for the plot
    summary = pandas.DataFrame(columns=["has_gpu", "language"])
    for repo, data in meta.items():
        df.loc[repo] = [data["has_gpu"], data["language"]]
        if data["language"] not in filtered_set:
            continue
        if data["has_gpu"]:
            summary.loc[repo] = ["yes", data["language"]]
        else:
            summary.loc[repo] = ["no", data["language"]]

    df.to_csv("repos-with-gpus.csv")

    # df = pandas.read_csv('repos-with-gpus.csv', index_col=0)
    sns.histplot(summary, x="language", hue="has_gpu", multiple="stack")
    plt.title("Research Software using GPU (N=5.5K)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("repos-with-gpus-stacked.png")
    plt.close()

    sns.histplot(summary, x="language", hue="has_gpu", multiple="dodge")
    plt.title("Research Software using GPU (N=5.5K)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("repos-with-gpus.png")
    plt.close()
    summary.to_csv("repos-with-gpus-greater-than-50.csv")


if __name__ == "__main__":
    main()
