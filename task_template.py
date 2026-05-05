import os
import sys
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import requests
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from scipy.stats import norm
from sklearn.metrics import roc_curve
from tqdm import tqdm

BASE         = Path(__file__).parent
PUB_PATH     = BASE / "pub.pt"
PRIV_PATH    = BASE / "priv.pt"
MODEL_PATH   = BASE / "model.pt"
OUTPUT_CSV   = BASE / "submission.csv"
SHADOW_DIR   = BASE / "shadow_models"
SCORES_DIR   = BASE / "scores"


BASE_URL     = "http://34.63.153.158"
API_KEY      = "793942ed492393f97fec0dd0bc7ce188"
TASK_ID      = "01-mia"


NUM_SHADOW   = 64
EPOCHS       = 100
BATCH_SIZE   = 128
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES  = 9


MEAN         = [0.7406, 0.5331, 0.7059]
STD          = [0.1491, 0.1864, 0.1301]


DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


SHADOW_DIR.mkdir(exist_ok=True)
SCORES_DIR.mkdir(exist_ok=True)


print(f"Device : {DEVICE}")


class TaskDataset(Dataset):
   def __init__(self, transform=None):
       self.ids       = []
       self.imgs      = []
       self.labels    = []
       self.transform = transform


   def __getitem__(self, index):
       id_   = self.ids[index]
       img   = self.imgs[index]
       if self.transform is not None:
           img = self.transform(img)
       label = self.labels[index]
       return id_, img, label


   def __len__(self):
       return len(self.ids)




class MembershipDataset(TaskDataset):
   def __init__(self, transform=None):
       super().__init__(transform)
       self.membership = []


   def __getitem__(self, index):
       id_, img, label = super().__getitem__(index)
       return id_, img, label, self.membership[index]


train_transform = transforms.Compose([
   transforms.Resize(32),
   transforms.RandomHorizontalFlip(),
   transforms.RandomCrop(32, padding=4),
   transforms.Normalize(mean=MEAN, std=STD),
])


eval_transform = transforms.Compose([
   transforms.Resize(32),
   transforms.Normalize(mean=MEAN, std=STD),
])



HF_BASE = "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main"


def download_if_missing():
   for name, path in [("pub.pt", PUB_PATH), ("priv.pt", PRIV_PATH), ("model.pt", MODEL_PATH)]:
       if path.exists():
           print(f"  [✓] {name} already exists")
       else:
           print(f"  [↓] Downloading {name} ...")
           subprocess.run(["wget", "-q", "-O", str(path), f"{HF_BASE}/{name}"], check=True)
           print(f"  [✓] {name} saved")


def make_model():
   m = resnet18(weights=None)
   m.conv1   = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
   m.maxpool = nn.Identity()
   m.fc      = nn.Linear(512, NUM_CLASSES)
   return m.to(DEVICE)




def load_model(path):
   m = make_model()
   m.load_state_dict(torch.load(path, map_location=DEVICE))
   m.eval()
   return m



def collate_skip_none(batch):
   cleaned = []
   for sample in batch:
       if len(sample) == 4 and sample[3] is None:
           cleaned.append(sample[:3])
       else:
           cleaned.append(sample)
   return torch.utils.data.dataloader.default_collate(cleaned)


def get_logit_scores(model, loader):
   scores = []
   model.eval()
   with torch.no_grad():
       for batch in loader:
           _, imgs, labels = batch[:3]
           imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
           logits  = model(imgs)
           correct = logits[torch.arange(len(labels)), labels]
           scores.append(correct.cpu().numpy())
   return np.concatenate(scores)


def yeom_scores(model, loader):
   scores = []
   criterion = nn.CrossEntropyLoss(reduction='none')
   model.eval()
   with torch.no_grad():
       for batch in loader:
           _, imgs, labels = batch[:3]
           imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
           logits = model(imgs)
           loss   = criterion(logits, labels)      
           scores.append(-loss.cpu().numpy())       
   return np.concatenate(scores)


def mentr_scores(model, loader):
   scores = []
   model.eval()
   with torch.no_grad():
       for batch in loader:
           _, imgs, labels = batch[:3]
           imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
           probs  = torch.softmax(model(imgs), dim=1)              # (B, C)
           p_y    = probs[torch.arange(len(labels)), labels]       # (B,)
           entr   = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (B,)
           mentr  = -(1 - p_y) * entr
           scores.append(mentr.cpu().numpy())
   return np.concatenate(scores)



def collect_shadow_scores(pub_ds, priv_ds, pub_loader, priv_loader):
   scores_pub_path  = SCORES_DIR / "scores_pub_shadow.npy"
   scores_priv_path = SCORES_DIR / "scores_priv_shadow.npy"
   masks_path       = SCORES_DIR / "masks.npy"


   N_pub  = len(pub_ds)
   N_priv = len(priv_ds)


   if scores_pub_path.exists() and scores_priv_path.exists() and masks_path.exists():
       print("  Loading cached shadow scores...")
       return (np.load(scores_pub_path),
               np.load(scores_priv_path),
               np.load(masks_path))


   shadow_paths = list(SHADOW_DIR.glob("shadow_*.pt"))
   if len(shadow_paths) < NUM_SHADOW:
       print(f"  Only {len(shadow_paths)}/{NUM_SHADOW} shadow models found — skipping LiRA")
       return None, None, None


   scores_pub  = np.zeros((N_pub,  NUM_SHADOW), dtype=np.float32)
   scores_priv = np.zeros((N_priv, NUM_SHADOW), dtype=np.float32)


   for j in tqdm(range(NUM_SHADOW), desc="  Querying shadow models"):
       shadow = load_model(SHADOW_DIR / f"shadow_{j:04d}.pt")
       scores_pub[:, j]  = get_logit_scores(shadow, pub_loader)
       scores_priv[:, j] = get_logit_scores(shadow, priv_loader)
       del shadow
       if DEVICE == "cuda":
           torch.cuda.empty_cache()


   masks = np.zeros((N_pub, NUM_SHADOW), dtype=np.uint8)
   for j in range(NUM_SHADOW):
       masks[:, j] = np.load(SHADOW_DIR / f"mask_{j:04d}.npy")


   np.save(scores_pub_path,  scores_pub)
   np.save(scores_priv_path, scores_priv)
   np.save(masks_path, masks)


   return scores_pub, scores_priv, masks




def compute_lira_scores(shadow_scores, target_scores, masks=None):
   N        = shadow_scores.shape[0]
   z_scores = np.zeros(N, dtype=np.float32)
   for i in range(N):
       out_scores = shadow_scores[i, masks[i] == 0] if masks is not None else shadow_scores[i]
       if len(out_scores) < 5:
           continue
       mu          = out_scores.mean()
       sigma       = out_scores.std() + 1e-8
       z_scores[i] = (target_scores[i] - mu) / sigma
   return z_scores


def normalize(x):
   return (x - x.min()) / (x.max() - x.min() + 1e-8)




def tpr_at_5fpr(membership, scores):
   fpr, tpr, _ = roc_curve(membership, scores)
   return float(tpr[fpr <= 0.05][-1])


def main():


   print("\n── Step 0: Downloading data")
   download_if_missing()


   print("\nLoading datasets...")
   pub_ds  = torch.load(PUB_PATH,  weights_only=False)
   priv_ds = torch.load(PRIV_PATH, weights_only=False)
   print(f"  pub.pt:  {len(pub_ds)} samples")
   print(f"  priv.pt: {len(priv_ds)} samples")


   pub_ds.transform  = eval_transform
   priv_ds.transform = eval_transform


   pub_loader  = DataLoader(pub_ds,  batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True,
                            collate_fn=collate_skip_none)
   priv_loader = DataLoader(priv_ds, batch_size=256, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=collate_skip_none)


   print("\nLoading target model...")
   target_model   = load_model(MODEL_PATH)
   pub_membership = np.array(pub_ds.membership)


   print("\n── Comparing attacks on pub.pt ")


   logit_pub  = get_logit_scores(target_model, pub_loader)
   yeom_pub   = yeom_scores(target_model, pub_loader)
   mentr_pub  = mentr_scores(target_model, pub_loader)


   results = {}
   for name, scores in [("logit",  logit_pub),
                         ("yeom",   yeom_pub),
                         ("mentr",  mentr_pub)]:
       t = tpr_at_5fpr(pub_membership, scores)
       results[name] = t
       print(f"  {name:8s}  TPR@5%FPR: {t:.4f}")


   ens_pub = (normalize(logit_pub) + normalize(yeom_pub) + normalize(mentr_pub)) / 3
   t_ens   = tpr_at_5fpr(pub_membership, ens_pub)
   results["ensemble"] = t_ens
   print(f"  {'ensemble':8s}  TPR@5%FPR: {t_ens:.4f}")


   shadow_pub, shadow_priv, masks = collect_shadow_scores(
       pub_ds, priv_ds, pub_loader, priv_loader)


   target_pub_logit  = get_logit_scores(target_model, pub_loader)
   target_priv_logit = get_logit_scores(target_model, priv_loader)


   lira_pub = None
   if shadow_pub is not None:
       lira_pub  = compute_lira_scores(shadow_pub,  target_pub_logit,  masks=masks)
       t_lira    = tpr_at_5fpr(pub_membership, lira_pub)
       results["lira"] = t_lira
       print(f"  {'lira':8s}  TPR@5%FPR: {t_lira:.4f}")


       ens_with_lira = (normalize(logit_pub) + normalize(yeom_pub) +
                        normalize(mentr_pub) + normalize(lira_pub)) / 4
       t_ens_lira = tpr_at_5fpr(pub_membership, ens_with_lira)
       results["ens+lira"] = t_ens_lira
       print(f"  {'ens+lira':8s}  TPR@5%FPR: {t_ens_lira:.4f}")


   best_name = max(results, key=results.get)
   print(f"\n  ✓ Best attack: {best_name}  (TPR@5%FPR = {results[best_name]:.4f})")


   print("\n── Scoring priv.pt with best attack")


   logit_priv = get_logit_scores(target_model, priv_loader)
   yeom_priv  = yeom_scores(target_model, priv_loader)
   mentr_priv = mentr_scores(target_model, priv_loader)


   if best_name == "logit":
       priv_raw = logit_priv
   elif best_name == "yeom":
       priv_raw = yeom_priv
   elif best_name == "mentr":
       priv_raw = mentr_priv
   elif best_name == "ensemble":
       priv_raw = (normalize(logit_priv) + normalize(yeom_priv) + normalize(mentr_priv)) / 3
   elif best_name == "lira":
       lira_priv = compute_lira_scores(shadow_priv, target_priv_logit, masks=None)
       priv_raw  = lira_priv
   elif best_name == "ens+lira":
       lira_priv = compute_lira_scores(shadow_priv, target_priv_logit, masks=None)
       priv_raw  = (normalize(logit_priv) + normalize(yeom_priv) +
                    normalize(mentr_priv) + normalize(lira_priv)) / 4


   priv_scores = normalize(priv_raw).astype(np.float32)


   print("\n── Writing submission.csv")
   priv_ids = [str(i.item() if isinstance(i, torch.Tensor) else i)
               for i in priv_ds.ids]


   df = pd.DataFrame({"id": priv_ids, "score": priv_scores})


   assert len(df) == len(priv_ds),        "Row count mismatch"
   assert df["id"].nunique() == len(df),  "Duplicate IDs"
   assert df["score"].between(0,1).all(), "Scores out of [0,1]"
   assert df["score"].notna().all(),      "NaN scores found"


   df.to_csv(OUTPUT_CSV, index=False)
   print(f"  ✓ Saved {len(df)} rows → {OUTPUT_CSV}")
   print(f"  ✓ Attack used: {best_name}  |  pub TPR@5%FPR: {results[best_name]:.4f}")




main()


# submit
def die(msg):
   print(msg, file=sys.stderr)
   sys.exit(1)


parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()


submit_path = OUTPUT_CSV


if not submit_path.exists():
   die(f"File not found: {submit_path}")


try:
   with open(submit_path, "rb") as f:
       resp = requests.post(
           f"{BASE_URL}/submit/{TASK_ID}",
           headers={"X-API-Key": API_KEY},
           files={"file": (submit_path.name, f, "application/csv")},
           timeout=(10, 600),
       )
   try:
       body = resp.json()
   except Exception:
       body = {"raw_text": resp.text}


   if resp.status_code == 413:
       die("Upload rejected: file too large (HTTP 413).")


   resp.raise_for_status()


   print("Successfully submitted.")
   print("Server response:", body)
   submission_id = body.get("submission_id")
   if submission_id:
       print(f"Submission ID: {submission_id}")


except requests.exceptions.RequestException as e:
   detail = getattr(e, "response", None)
   print(f"Submission error: {e}")
   if detail is not None:
       try:
           print("Server response:", detail.json())
       except Exception:
           print("Server response (text):", detail.text)
   sys.exit(1)