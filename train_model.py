import numpy as np
import pandas as pd
import cv2
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from skimage.feature import local_binary_pattern, hog
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# ======================
# CONFIG
# ======================
DATA_CSV      = "HAM10000_metadata.csv"
IMG_SIZE      = 128
IMAGE_FOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
CACHE_X       = "features_X_real.npy"
CACHE_Y       = "features_y_real.npy"

# ── Per-class augmentation multiplier ──
# Calculated to bring every class to ~1000-2000 samples
# so class weights end up close to 1.0 for all classes
#
#   df    115  × 20 = 2300   ✅ perfect
#   vasc  142  × 14 = 1988   ✅ perfect
#   akiec 327  ×  9 = 2943   ⬆ increased from 3
#   bcc   514  ×  5 = 2570   ⬆ increased from 2
#   mel   1113 ×  2 = 2226   ⬆ increased from 0
#   bkl   1099 ×  2 = 2198   ⬆ increased from 0
#   nv    6705 ×  0 = 6705   class weight handles dominance
#
# Set to 0 to skip augmentation for that class
AUG_TARGET = {
    'df':    20,
    'vasc':  14,
    'akiec':  9,
    'bcc':    5,
    'mel':    2,
    'bkl':    2,
    'nv':     0,
}

CLASS_NAMES = {
    0: 'Actinic Keratoses',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular Lesions'
}

# ======================
# LOAD DATA
# ======================
metadata = pd.read_csv(DATA_CSV)
le = LabelEncoder()
metadata['label'] = le.fit_transform(metadata['dx'])

print("📊 DATASET INFO")
print(f"Total images: {len(metadata)}")
print("\nReal class distribution:")
for cls, count in metadata['dx'].value_counts().items():
    print(f"  {cls:6s}  {count:5d}")

# ======================
# IMAGE FINDER
# ======================
def find_image(image_id):
    for folder in IMAGE_FOLDERS:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

# ======================
# AUGMENTATION
# Creates realistic variations of an image
# Only applied to minority class images
# ======================
def augment_image(img, aug_index):
    """
    Generates one augmented version of an image.
    aug_index controls which combination of transforms to apply
    so each of the AUG_MULTIPLIER copies looks different.

    Transforms used:
    - Horizontal flip    (mirrors left-right)
    - Vertical flip      (mirrors top-bottom)
    - 90/180/270 rotation
    - Brightness shift   (simulate different lighting)
    - Gaussian blur      (simulate slight focus variation)
    - Contrast adjustment
    """
    img = img.copy()

    # -- Flip --
    if aug_index % 2 == 0:
        img = cv2.flip(img, 1)          # horizontal flip
    if aug_index % 3 == 0:
        img = cv2.flip(img, 0)          # vertical flip

    # -- Rotation (0, 90, 180, 270) --
    angles = [0, 90, 180, 270]
    angle  = angles[aug_index % 4]
    if angle == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # -- Brightness shift --
    # Shifts pixel values up or down slightly to simulate lighting changes
    brightness_shift = [-30, -15, 0, 15, 30][aug_index % 5]
    img = np.clip(img.astype(np.int32) + brightness_shift, 0, 255).astype(np.uint8)

    # -- Slight blur on some copies --
    if aug_index % 3 == 1:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # -- Contrast adjustment --
    if aug_index % 2 == 1:
        # alpha > 1 increases contrast, < 1 decreases
        alpha = 1.2 if aug_index % 4 < 2 else 0.85
        img   = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    return img

# ======================
# FEATURES
# ======================
def extract_color_histogram(img):
    f = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [32], [0, 256])
        f.extend(cv2.normalize(hist, hist).flatten())
    return f

def extract_hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [32], [0, 256])
        f.extend(cv2.normalize(hist, hist).flatten())
    return f

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, 24, 3, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
    return hist

def extract_glcm(img):
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(gray, [1], [0], 8, True, True)
    return [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0]
    ]

def extract_hog_feat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def extract_color_moments(img):
    f = []
    for i in range(3):
        ch   = img[:, :, i].astype(float)
        mean = np.mean(ch)
        std  = np.std(ch)
        skew = np.mean(((ch - mean) / (std + 1e-7)) ** 3)
        f.extend([mean, std, skew])
    return f

def extract_asymmetry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w   = mask.shape
    top    = mask[:h//2, :]
    bottom = np.flipud(mask[h//2:, :])
    left   = mask[:, :w//2]
    right  = np.fliplr(mask[:, w//2:])
    min_h  = min(top.shape[0], bottom.shape[0])
    min_w  = min(left.shape[1], right.shape[1])
    h_diff = np.sum(np.abs(top[:min_h].astype(int) - bottom[:min_h].astype(int))) / (h * w * 255)
    v_diff = np.sum(np.abs(left[:, :min_w].astype(int) - right[:, :min_w].astype(int))) / (h * w * 255)
    return [h_diff, v_diff]

def extract_border_irregularity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0]
    cnt         = max(contours, key=cv2.contourArea)
    area        = cv2.contourArea(cnt)
    perimeter   = cv2.arcLength(cnt, True)
    compactness = (perimeter ** 2) / (area + 1e-7)
    return [compactness]

def extract_color_variance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return [
        np.std(hsv[:, :, 0]),
        np.std(hsv[:, :, 1]),
        np.mean(hsv[:, :, 0])
    ]

def extract_features(img):
    return np.concatenate([
        extract_color_histogram(img),
        extract_hsv_histogram(img),
        extract_lbp(img),
        extract_glcm(img),
        extract_hog_feat(img),
        extract_color_moments(img),
        extract_asymmetry(img),
        extract_border_irregularity(img),
        extract_color_variance(img)
    ]).astype(np.float32)

# ======================
# PROCESS ONE ROW
# Returns:
#   - 1 real sample always
#   - AUG_TARGET[dx] extra augmented samples for minority classes
#   - 0 extra samples for majority classes (mel, bkl, nv)
# ======================
def process(row):
    path = find_image(row['image_id'])
    if not path:
        return []

    img = cv2.imread(path)
    if img is None:
        return []

    img   = preprocess(img)
    label = row['label']
    dx    = row['dx']

    # Always extract features from the real image
    samples = [(extract_features(img), label)]

    # ── Per-class augmentation ──
    # Each class gets a different number of augmented copies
    # based on how rare it is — so all classes end up ~equal size
    # The label stays the same because it's the same disease,
    # just photographed from a different angle / lighting
    multiplier = AUG_TARGET.get(dx, 0)
    for i in range(multiplier):
        aug_img  = augment_image(img, aug_index=i)
        aug_feat = extract_features(aug_img)
        samples.append((aug_feat, label))

    return samples

# ======================
# FEATURE EXTRACTION
# ======================
if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print("\n⚡ Loading cached features...")
    X = np.load(CACHE_X)
    y = np.load(CACHE_Y)
    print(f"Shape: {X.shape}")
else:
    print("\n🔄 Extracting features + augmenting minority classes...")
    print(f"   Per-class augmentation targets: {AUG_TARGET}")

    # process() now returns a LIST of (features, label) tuples per row
    # (1 real + up to AUG_MULTIPLIER augmented)
    all_results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process)(row) for _, row in metadata.iterrows()
    )

    # Flatten the list of lists into one flat list
    flat = [item for sublist in all_results for item in sublist]

    X = np.array([r[0] for r in flat], dtype=np.float32)
    y = np.array([r[1] for r in flat])

    np.save(CACHE_X, X)
    np.save(CACHE_Y, y)
    print(f"\nExtracted {len(X)} total samples | Features per sample: {X.shape[1]}")

# ======================
# SHOW AUGMENTED CLASS DISTRIBUTION
# ======================
print("\n📊 Class distribution AFTER augmentation:")
vals, cnts = np.unique(y, return_counts=True)
for v, c in zip(vals, cnts):
    cls_name = le.inverse_transform([v])[0]
    print(f"  {cls_name:6s}  {c:5d}")

# ======================
# SPLIT — stratified
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\n📊 TRAIN-TEST SPLIT")
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ======================
# SCALE
# ======================
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)         # transform only — never fit on test

# ======================
# SMOTE OVERSAMPLING
# ======================
print("\n⚖️  Applying SMOTE to perfectly balance classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\n📊 Class distribution AFTER SMOTE:")
vals, cnts = np.unique(y_train_res, return_counts=True)
for v, c in zip(vals, cnts):
    cls_name = le.inverse_transform([v])[0]
    print(f"  {cls_name:6s}  {c:5d}")

# ======================
# INDIVIDUAL MODELS
# ======================
print("\n🚀 Training individual models...")

# 1. XGBoost
try:
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, 
        subsample=0.8, colsample_bytree=0.8, device='cuda', verbosity=0
    )
    xgb_model.fit(np.random.rand(10, 5), list(range(10))) # Test CUDA
    print("\n⚡ XGBoost using GPU (CUDA)")
except Exception:
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, 
        subsample=0.8, colsample_bytree=0.8, tree_method='hist', verbosity=0
    )
    print("\n💻 XGBoost using CPU")

xgb_model.fit(X_train_res, y_train_res)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
print(f"✅ XGBoost Accuracy:             {xgb_acc*100:.2f}%")

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"✅ Random Forest Accuracy:       {rf_acc*100:.2f}%")

# 3. HistGradientBoosting
hgb_model = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=42)
hgb_model.fit(X_train_res, y_train_res)
hgb_acc = accuracy_score(y_test, hgb_model.predict(X_test))
print(f"✅ HistGradientBoosting Accuracy: {hgb_acc*100:.2f}%")

# ======================
# ENSEMBLE MODEL
# ======================
print("\n🚀 Combining into Soft Voting Ensemble...")
ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model), ('hgb', hgb_model)],
    voting='soft',
    n_jobs=1
)

ensemble_model.fit(X_train_res, y_train_res)
ens_acc = accuracy_score(y_test, ensemble_model.predict(X_test))
print(f"🏆 ENSEMBLE Overall Accuracy:    {ens_acc*100:.2f}%")

# ======================
# EVALUATE
# ======================
print("\n📋 Ensemble Classification Report:")
target_names = [le.inverse_transform([i])[0] for i in sorted(np.unique(y_test))]
print(classification_report(
    y_test, ensemble_model.predict(X_test),
    target_names=target_names
))

# ======================
# SAVE
# ======================
accuracies_dict = {
    'XGBoost': round(xgb_acc * 100, 2),
    'Random Forest': round(rf_acc * 100, 2),
    'HistGradientBoosting': round(hgb_acc * 100, 2),
    'Ensemble ML (Overall)': round(ens_acc * 100, 2)
}

pickle.dump(ensemble_model,  open("skin_cancer_model.pkl", "wb"))
pickle.dump(scaler,          open("scaler.pkl",             "wb"))
pickle.dump(le,              open("label_encoder.pkl",      "wb"))
pickle.dump(accuracies_dict, open("model_accuracies.pkl", "wb"))

print("\n💾 Saved: skin_cancer_model.pkl  scaler.pkl  label_encoder.pkl  model_accuracies.pkl")