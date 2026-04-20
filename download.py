import os
import argparse
import zipfile
import gdown


gdrive_ids = {"mantis": "17Uj9EwyGGCk9w_LIqDjlTx1y4MU7xxPv",}# lien issus du github du papier

def download_and_extract_gdown(dataset_name):
    dataset_name = dataset_name
    
    if dataset_name not in gdrive_ids:
        print(f"Le dataset {dataset_name} n'est pas configuré")
        return

    drive_id = gdrive_ids[dataset_name]
    output_dir = f"data/{dataset_name}"
    zip_path = f"data/{dataset_name}.zip"

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, "train.tsv")):
        print(f"Les données existent déjà : {output_dir}.")
        return

    print(f"Téléchargement de {dataset_name}")
    gdown.download(id=drive_id, output=zip_path)

    if os.path.exists(zip_path):
        print(f"extraction : {dataset_name}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # on remove le zip 
        os.remove(zip_path)
    else:
        print("echec du telechargement")
        
# python download.py --dataset mantis
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset à télécharger (ex: mantis, qqp, trec)")
    args = parser.parse_args()
    
    download_and_extract_gdown(args.dataset)