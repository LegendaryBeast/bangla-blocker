#!/usr/bin/env python3
"""
Upload BanglaBlocker-Base to Hugging Face Hub
Usage:
    python3 upload_to_hf.py
"""
import os
import sys
from huggingface_hub import HfApi, create_repo

def main():
    print("=" * 60)
    print("  BanglaBlocker-Base -> Hugging Face Upload Tool")
    print("=" * 60)
    
    # 1. Ask for Hugging Face Username
    username = input("\n1. Enter your Hugging Face Username (default: tanzim-hasan): ").strip() or "tanzim-hasan"
        
    repo_name = input("2. Repository name (default: banglablocker-base): ").strip() or "banglablocker-base"
    repo_id = f"{username}/{repo_name}"
    
    # 2. Ask for Access Token
    token = input("3. Paste your Hugging Face WRITE Access Token (hf_...): ").strip()
    if not token.startswith("hf_"):
        print("Warning: Hugging Face tokens typically start with 'hf_' - proceeding anyway.")
        
    # 3. Choose Visibility
    visibility = input("4. Make repository Public or Private? [public/private] (default: public): ").strip().lower() or "public"
    is_private = (visibility == "private")
    
    local_dir = os.path.dirname(os.path.abspath(__file__)) + "/banglablocker-base"
    print(f"\nTarget Repository: https://huggingface.co/{repo_id}")
    print(f"Local Folder:      {local_dir}")
    print(f"Visibility:        {'Private' if is_private else 'Public'}")
    
    confirm = input("\nReady to upload? [Y/n]: ").strip().lower()
    if confirm not in ('', 'y', 'yes'):
        print("Upload cancelled.")
        sys.exit(0)
        
    api = HfApi(token=token)
    
    # 4. Create repository if it doesn't exist
    print(f"\n[1/2] Creating/verifying repository '{repo_id}' on Hugging Face...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", private=is_private, exist_ok=True)
        print("  -> Repository verified successfully!")
    except Exception as e:
        print(f"  -> Note/Error: {e}")
        
    # 5. Upload files
    print(f"\n[2/2] Uploading model files to https://huggingface.co/{repo_id} ...")
    print("  (This may take 1-3 minutes depending on internet connection speed)\n")
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial release of BanglaBlocker-Base checkpoint and model card"
        )
        print("\n" + "=" * 60)
        print(f"  SUCCESS! Your model is now live at:")
        print(f"  https://huggingface.co/{repo_id}")
        print("=" * 60)
    except Exception as e:
        print(f"\nUpload failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
