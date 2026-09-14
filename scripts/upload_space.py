"""Upload deploy/hf_space to a Hugging Face Space, creating the Space if needed.

Before running:
  1. Create a free account at https://huggingface.co
  2. Log in on this computer (you paste your own access token when asked):
         hf auth login
     (hf.exe is installed with your main Python 3.13, in its Scripts folder)
  3. Build the folder:  python -m scripts.build_space

Run:  python -m scripts.upload_space --space YOUR_USERNAME/malaria-smear-analyzer
      (add --private to make the Space visible only to you)
"""
import argparse

from huggingface_hub import HfApi

from scripts.build_space import SPACE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="username/space-name")
    parser.add_argument("--private", action="store_true", help="only you can see the Space")
    args = parser.parse_args()

    if not (SPACE_DIR / "Dockerfile").exists():
        raise SystemExit("Build the Space folder first: python -m scripts.build_space")

    api = HfApi()
    # whoami() fails with a clear message if you haven't logged in yet
    print(f"Logged in as: {api.whoami()['name']}")

    api.create_repo(args.space, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=SPACE_DIR, repo_id=args.space, repo_type="space",
                      commit_message="Upload Malaria Smear Analyzer")

    print(f"Uploaded. The Space builds in a few minutes: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
