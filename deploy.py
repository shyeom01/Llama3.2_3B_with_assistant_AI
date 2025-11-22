import os
from huggingface_hub import HfApi, create_repo, upload_file
import getpass

def deploy():
    print("---------------------------------------------------------")
    print("   Automated Deployment to Hugging Face Spaces 🚀")
    print("---------------------------------------------------------")
    
    # 1. Get Token
    print("\nPlease enter your Hugging Face Access Token (WRITE permission required).")
    print("You can find it here: https://huggingface.co/settings/tokens")
    token = getpass.getpass("HF Token: ").strip()
    
    if not token:
        print("Error: Token is required.")
        return

    api = HfApi(token=token)
    
    try:
        user = api.whoami()
        username = user['name']
        print(f"\nLogged in as: {username}")
    except Exception as e:
        print(f"Error: Invalid token. {e}")
        return

    # 2. Get Space Name
    print("\nEnter the name for your new Space (e.g., 'my-llama-chat').")
    space_name = input("Space Name: ").strip()
    repo_id = f"{username}/{space_name}"

    # 3. Create README.md locally
    print("\nGenerating README.md...")
    readme_content = f"""---
title: {space_name}
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# {space_name}

This is a Streamlit app deployed from the fine-tuned model project.
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 4. Create Space
    print(f"\nCreating Space: {repo_id}...")
    try:
        # Try creating without specifying SDK (letting README define it)
        # If that fails, we try with 'static' as a placeholder and overwrite it later
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                exist_ok=True
            )
        except Exception as e:
            if "sdk" in str(e).lower() or "invalid option" in str(e).lower():
                print("Standard creation failed, trying fallback method...")
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="space",
                    space_sdk="static",
                    exist_ok=True
                )
            else:
                raise e
                
        print("✅ Space created.")
    except Exception as e:
        print(f"Error creating Space: {e}")
        return

    # 5. Upload Files
    print("\nUploading files...")
    files_to_upload = ["README.md", "app.py", "requirements.txt"]
    
    for file in files_to_upload:
        if not os.path.exists(file):
            print(f"Warning: {file} not found. Skipping.")
            continue
            
        print(f"Uploading {file}...")
        try:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="space"
            )
        except Exception as e:
            print(f"Error uploading {file}: {e}")

    print("✅ Files uploaded.")

    # 6. Set Secret
    print("\nConfiguring Secrets...")
    try:
        # We use the same token for the secret so the app can access the gated model
        api.add_space_secret(
            repo_id=repo_id,
            key="HF_TOKEN",
            value=token
        )
        print("✅ Secret 'HF_TOKEN' set.")
    except Exception as e:
        print(f"Error setting secret: {e}")

    print("\n---------------------------------------------------------")
    print("🎉 Deployment Complete!")
    print(f"Your app is building at: https://huggingface.co/spaces/{repo_id}")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    deploy()
