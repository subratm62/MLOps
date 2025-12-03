import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "subratm62/Tourism-Package-Prediction"   # space repo
REPO_TYPE = "space"

api = HfApi(token=HF_TOKEN)

# Login to your Hugging Face account using your access token
login(token=HF_TOKEN)

# -----------------------------
# 1. Create Space if it doesn't exist
# -----------------------------
def ensure_space_exists(repo_id: str):
    try:
        api.repo_info(repo_id, repo_type="space")
        print(f"Space '{repo_id}' already exists!")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating it now...")

        create_repo(
            repo_id.split("/")[1],   # extract repo name only
            repo_type="space",
            space_sdk="docker",           # Using Docker SDK
            private=False
        )
        print(f"Space '{repo_id}' created successfully!")

# Ensure the Space exists (create if missing)
ensure_space_exists(REPO_ID)

# -----------------------------
# 2. Upload folder to the Space
# -----------------------------
print("📤 Uploading files to HuggingFace Space...")

api.upload_folder(
    folder_path="tourism_project/deployment",   # local folder
    repo_id=REPO_ID,
    repo_type="space",
    path_in_repo="",    # root of the Space repo
)

print("✅ Upload complete!")
print(f"🔗 Visit your Space at: https://huggingface.co/spaces/{REPO_ID}")

