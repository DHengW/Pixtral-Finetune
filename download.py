from huggingface_hub import snapshot_download
model_dir = snapshot_download(repo_id = 'mistral-community/pixtral-12b',local_dir='data/mistral-community/pixtral-12b')
model_dir = snapshot_download(repo_id = 'teamix-aicrowd/index_aug_image_rerank_dataset', repo_type='dataset',local_dir='data/teamix-aicrowd/index_aug_image_rerank_dataset',token='')