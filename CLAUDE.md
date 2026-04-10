# CLAUDE.md — PERFECT 10/10 VERSION (HYBRID TREK RECOMMENDER)

You are an expert machine learning engineer specializing in recommendation systems.
Build a complete **hybrid trek recommendation system in Python**.

Write **clean, well-commented, production-ready code**.
The script must run end-to-end using:

```bash
python train.py
```

Use ONLY allowed libraries. No shortcuts. No placeholders.

---

# GLOBAL RULES

- Set reproducibility:

```python
np.random.seed(42)
```

- DO NOT use:
  - surprise, implicit, lightfm, tensorflow, torch
  - sklearn cosine similarity helpers

- MUST implement:
  - ALS from scratch
  - Cosine similarity manually using numpy

---

# DATASET

Load from:

```
final_trek_dataset_with_images.json
```

### Cleaning rules:

1. Duplicate fix:
   - If `trek_id='puh001'` appears twice:
     - First stays `puh001`
     - Second becomes `puh002`

2. Missing values:
   - `base_camp_altitude_m` → fill with **median**

---

# USER GENERATION

Generate exactly **600 users**

Structure:

```python
{
  "user_id": "user_0001",
  "preferences": {
    "difficulty": int (1–6),
    "budget_max": int (200–5000),
    "duration_max": int (1–30),
    "fitness": int (1–4)
  }
}
```

Use 15 base profiles × 40 users each.

Apply noise:

- difficulty ±1 → clamp [1,6]
- budget × uniform(0.75,1.35)
- duration ±2 → clamp [1,30]
- fitness ±1 → clamp [1,4]

---

# INTERACTIONS (CRITICAL)

Generate interactions for **45% of user–trek pairs**

### Schema:

```python
interaction = {
  "user_id": str,
  "trek_id": str,
  "views": int (0–10),
  "booked": bool,
  "favorites": bool,
  "rating": float (0 or 1–5),
  "time_spent_seconds": int (0–600)
}
```

---

## BASE SCORE

Compute using weighted sum:

- difficulty (3.0)
- budget (2.5)
- duration (2.0)
- fitness (2.0)
- trek rating (1.0 scaled 4.2–5.0 → 0–5)

---

## INTERACTION VALUES

```python
views = clamp(int(base_score * 2 + normal(0,0.5)), 0, 10)

booked = base_score > 4.2 and random() < 0.3

favorites = base_score > 3.5 and random() < 0.4

rating = round(base_score + normal(0,0.2), 1) if base_score > 3.0 else 0.0
rating = clamp(rating, 1.0, 5.0) if rating > 0 else 0.0

time_spent_seconds = clamp(int(base_score * 100 + normal(0,20)), 0, 600)
```

---

## WEIGHT FORMULA (MANDATORY)

```python
weight_raw = (
    views * 0.3 +
    (3.0 if booked else 0) +
    (1.5 if favorites else 0) +
    (rating / 5.0) * 2.0 +
    (time_spent_seconds / 600)
)
```

---

## NORMALIZATION (STRICT RULE)

Use **Min-Max scaling across ALL interactions**:

```python
weight = 0.5 + (weight_raw - min_w) * (5.0 - 0.5) / (max_w - min_w)
```

Then:

```python
weight += normal(0, 0.1)
weight = clip(weight, 0.5, 5.0)
```

---

# TREK FEATURE ENCODING

Use EXACT mappings.

Build matrix (24 columns):

```
difficulty, fitness, risk, altitude_risk, water, food, network,
permits, guide, evacuation, duration_days, distance_km,
max_altitude_m, altitude_gain_m, daily_hours, cost_min, cost_max,
temp_min, temp_max, nearest_medical_km, avg_rating, popularity,
group_min, group_max
```

Scale using:

```python
MinMaxScaler → [0,1]
```

Save scaler.

---

# ALS (FROM SCRATCH)

Matrix: users × treks using **weight**

Params:

```python
n_factors=25
n_iterations=25
lambda_reg=0.1
```

Print training RMSE every 5 iterations.

---

# CONTENT-BASED FILTERING

## Cosine similarity (MANDATORY IMPLEMENTATION)

```python
cos_sim(A, B) = (A @ B) / (||A|| * ||B||)
```

Vectorized implementation required.

---

## USER VECTOR

### New user:

Use preference mapping:

- difficulty → scaled
- fitness → scaled
- duration → scaled
- budget → map to cost_mid = (cost_min + cost_max)/2

### Returning user:

Use:

```python
profile_vector = weighted mean of trek_feature_scaled
```

---

# HYBRID MODEL

```python
alpha = min(0.85, total_weight_sum / 30.0)
```

Normalize BOTH:

- ALS scores → [0,5]
- CBF scores → [0,5]

```python
final_score = alpha * ALS + (1 - alpha) * CBF
```

---

# FILTERING RULE (STRICT)

Always exclude:

```python
treks already interacted with by user
```

Applies to:

- ALS
- CBF
- Hybrid

---

# METHODS

Implement:

1. recommend_from_preferences
2. recommend_als
3. recommend_hybrid
4. recommend_for_new_user (alias)

---

# EVALUATION

Print:

- RMSE
- MAE

Interpretation:

- <0.4 → Excellent
- <0.6 → Good
- <0.8 → Acceptable
- ≥0.8 → Needs tuning

---

# MODEL SAVE

Save:

```
models/recommender.pkl
```

Include ALL required keys.

---

# VISUALS

Save:

```
als_training_results.png
```

4 plots:

1. Training RMSE
2. Predicted vs Actual
3. Interaction weight distribution
4. Top recommended treks

---

# FINAL OUTPUT

The script must:

- Run fully
- Print all steps
- Save model + plots
- Show recommendation results

NO placeholders. NO missing functions.

---

# GOAL

Build a **production-grade hybrid recommendation system**
combining:

- ALS (collaborative filtering)
- Content-based filtering
- Behavioral interaction signals
- Adaptive blending
