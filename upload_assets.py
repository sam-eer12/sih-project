import os
import vercel_blob
from dotenv import load_dotenv

# Load the .env.local file created by the Vercel CLI
load_dotenv('.env.local')

# --- List of all files and folders to upload ---
# We will walk through the dataset directory to get all files
files_to_upload = [
    "server/phylum_classifier_model.pkl",
    "models/trained_models.pkl",
    "models/processed_data.pkl",
    "models/combined_features.npy",
    "test_models/trained_models.pkl",
    "test_models/processed_data.pkl",
    "test_models/combined_features.npy",
]

# Add all files from the dataset directory
dataset_root = "dataset"
for dirpath, _, filenames in os.walk(dataset_root):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)
        files_to_upload.append(full_path)

print("--- Starting Asset Upload to Vercel Blob ---")

# Dictionary to store the new URLs
uploaded_urls = {}

for file_path in files_to_upload:
    # The pathname in the blob store will be the same as the local path
    blob_pathname = file_path.replace("\\", "/") # Ensures forward slashes for the path
    try:
        with open(file_path, 'rb') as f:
            print(f"Uploading {file_path} to {blob_pathname}...")
            blob_result = vercel_blob.put(blob_pathname, f.read())

            # Store the URL for later use
            uploaded_urls[blob_pathname] = blob_result['url']

            print(f"  -> Success! URL: {blob_result['url']}")

    except FileNotFoundError:
        print(f"  -> Error: File not found at {file_path}")
    except Exception as e:
        print(f"  -> An error occurred: {e}")

print("\n--- Upload Complete! ---")
print("Save these URLs. You will need them for Step 3.")

# Print a summary of URLs for easy copy-pasting
for path, url in uploaded_urls.items():
    print(f'"{path}": "{url}",')