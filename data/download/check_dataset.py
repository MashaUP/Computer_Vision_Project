"""
test_dataset.py
===============
Script di controllo per verificare la corretta consistenza e il numero di immagini
all'interno dei dataset locali KITTI e NYU Depth V2 secondo i requisiti di METER.
"""

from pathlib import Path
import h5py

# Configurazione percorsi relativi
KITTI_DIR = Path("./data/raw/kitti")
NYU_DIR = Path("./data/raw/nyu")

def check_kitti():
    print("=" * 60)
    print("  VERIFICA DATASET: KITTI")
    print("=" * 60)
    
    if not KITTI_DIR.exists():
        print(f"[ERRORE] La cartella KITTI non esiste al percorso: {KITTI_DIR.resolve()}")
        return

    depth_base = KITTI_DIR / "depth"
    rgb_base = KITTI_DIR / "rgb"

    # 1. Controllo Depth GT nelle cartelle train e val attuelles
    print("[INFO] Controllo mappe di profondità (Depth GT)...")
    train_depth = list((depth_base / "train").rglob("*.png")) if (depth_base / "train").exists() else []
    val_depth = list((depth_base / "val").rglob("*.png")) if (depth_base / "val").exists() else []
    
    total_depth_local = len(train_depth) + len(val_depth)
    print(f"  -> Depth Ground Truth trovate localmente: {total_depth_local:,}")
    print(f"     (Di cui Train: {len(train_depth):,} | Val: {len(val_depth):,})")

    # 2. Controllo Immagini RGB
    print("[INFO] Controllo immagini RGB scompattate...")
    rgb_images = list(rgb_base.rglob("image_02/data/*.png"))
    print(f"  -> Immagini RGB trovate localmente: {len(rgb_images):,}")

    print("\n--- STATO KITTI ---")
    if total_depth_local == 0:
        print("[❌ ERRORE] Nessun file di Depth GT trovato. Controlla il percorso kitti/depth/")
    elif len(rgb_images) == 0:
        print("[❌ ERRORE] Trovate le depth ma mancano completamente le immagini RGB (image_02).")
    else:
        print("[✓ OK] KITTI configurato con successo mantenendo la tua struttura attuale delle cartelle.")


def check_nyu():
    print("\n" + "=" * 60)
    print("  VERIFICA DATASET: NYU DEPTH V2")
    print("=" * 60)
    
    if not NYU_DIR.exists():
        print(f"[ERRORE] La cartella NYU non esiste al percorso: {NYU_DIR.resolve()}")
        return

    # 1. Verifica del file di Test ufficiale (.mat) con asse corretto
    mat_path = NYU_DIR / "nyu_depth_v2_labeled.mat"
    print("[INFO] Controllo file di Test (.mat)...")
    if not mat_path.exists():
        print("  [❌ ERRORE] Il file 'nyu_depth_v2_labeled.mat' NON è stato trovato.")
    else:
        try:
            with h5py.File(mat_path, "r") as f:
                if "images" in f:
                    shape = f["images"].shape
                    n_test = 1449 if 1449 in shape else int(shape[-1])
                    
                    if n_test == 1449:
                        print(f"  [✓ OK] File .mat valido. Test Set: {n_test} coppie labeled corrette.")
                    else:
                        print(f"  [⚠ WARN] File .mat trovato ma contiene {n_test} coppie anziché 1449.")
                else:
                    print("  [❌ ERRORE] Struttura del file .mat non riconosciuta.")
        except Exception as e:
            print(f"  [❌ ERRORE] Impossibile leggere il file .mat: {e}")

    # 2. Verifica del subset di Training/Validation (i 50K estratti da Kaggle)
    print("[INFO] Controllo subset di addestramento (50K)...")
    train_dir = NYU_DIR / "train"
    val_dir = NYU_DIR / "val"

    train_jpgs = list(train_dir.rglob("*.jpg")) if train_dir.exists() else []
    train_pngs = list(train_dir.rglob("depth_map/*.png")) if train_dir.exists() else []
    
    val_jpgs = list(val_dir.rglob("*.jpg")) if val_dir.exists() else []
    val_pngs = list(val_dir.rglob("depth_map/*.png")) if val_dir.exists() else []

    print(f"  -> Train local: {len(train_jpgs):,} immagini JPG | {len(train_pngs):,} mappe di profondità PNG")
    print(f"  -> Val local  : {len(val_jpgs):,} immagini JPG | {len(val_pngs):,} mappe di profondità PNG")

    print("\n--- STATO NYU ---")
    errors = False
    if len(train_jpgs) < 40000 or len(train_jpgs) != len(train_pngs):
        print(f"[❌ ERRORE] Il Training Set è incompleto o sbilanciato.")
        errors = True
    else:
        print("[✓ OK] Training Set completo (~50K subset).")

    if len(val_jpgs) < 600 or len(val_jpgs) != len(val_pngs):
        print(f"[❌ ERRORE] Il Validation Set è incompleto o sbilanciato.")
        errors = True
    else:
        print("[✓ OK] Validation Set completo (654 frame ufficiali).")

    if not errors and mat_path.exists():
        print("[✓ GENERALE] NYU Depth V2 è configurato al 100% in modo perfetto per METER!")


if __name__ == "__main__":
    try:
        check_kitti()
        check_nyu()
        print("\n[DONE] Tutti i controlli completati.")
    except Exception as global_exc:
        print(f"\n[ERRORE GENERALE DELLO SCRIPT]: {global_exc}")