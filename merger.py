import pickle

# Merge all batches
all_features = []
all_filenames = []

for i in range(1, 4):
    with open(f'embeddings_batch_{i}.pkl', 'rb') as f:
        all_features.extend(pickle.load(f))

    with open(f'filenames_batch_{i}.pkl', 'rb') as f:
        all_filenames.extend(pickle.load(f))

# Save merged files
with open('embeddings_merged.pkl', 'wb') as f:
    pickle.dump(all_features, f)

with open('filenames_merged.pkl', 'wb') as f:
    pickle.dump(all_filenames, f)

print("All batches merged and saved.")
