"""
Hybrid Trek Recommendation System (Enhanced)
=============================================
Combines ALS (collaborative filtering), Content-Based Filtering,
and adaptive blending into a production-grade hybrid recommender.

Enhanced with practical trekking parameters used as a post-ranking
layer so that the core ALS + CBF pipeline continues to surface
popular, well-known treks while the new attributes refine results
for users who care about specific logistics.

New trek metadata (stored but NOT injected into the feature matrix):
  - Seasonality ratings per season (1-5)
  - Altitude sickness detail (risk, acclimatization days, highest pass)
  - Permit details (types, cost, advance booking days)
  - Accommodation (types available, quality rating)
  - Transportation (drive time from Kathmandu, flight required)
  - Health & Safety (medical posts, helicopter evac, oxygen)

New user preferences:
  - preferred_season, ams_concern_level, permit_willingness,
    accommodation_preference

Usage:
    python train.py
"""

import json
import os
import pickle
import warnings
from statistics import median

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 : Load & Clean Trek Data
# ════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  STEP 1 : Loading & Cleaning Trek Data")
print("=" * 65)

with open('dataset/final_trek_dataset_with_provinces.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

treks = raw['treks']

# Fix duplicate trek_id: first puh001 stays, second becomes puh002
seen = set()
for t in treks:
    if t['trek_id'] in seen:
        if t['trek_id'] == 'puh001':
            t['trek_id'] = 'puh002'
    seen.add(t['trek_id'])

# Fill missing base_camp_altitude_m with median
altitudes = [t['base_camp_altitude_m'] for t in treks if t.get('base_camp_altitude_m') is not None]
med_alt = median(altitudes)
for t in treks:
    if t.get('base_camp_altitude_m') is None:
        t['base_camp_altitude_m'] = med_alt

n_treks = len(treks)
print(f"  Loaded {n_treks} treks")
print(f"  Median base camp altitude: {med_alt:.0f} m")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1b : Synthesize Enhanced Trek Attributes (metadata only)
# ════════════════════════════════════════════════════════════════════════════
#
#  IMPORTANT DESIGN DECISION:
#  These attributes are stored as metadata on each trek dict and used
#  ONLY in a lightweight post-ranking re-scorer. They are NOT added to
#  the 24-column feature matrix that drives ALS and CBF.
#
#  Why? The core 24 features come from real trek data and correlate with
#  trek popularity/quality. Injecting 14 randomly synthesized features
#  into the same matrix would dilute the signal from ratings, popularity,
#  difficulty, region, etc., causing the model to recommend obscure treks
#  instead of well-known, popular ones.
#
print("\n" + "=" * 65)
print("  STEP 1b : Synthesizing Enhanced Trek Attributes (metadata)")
print("=" * 65)

# Maps needed before synthesis
difficulty_map    = {'very easy': 1, 'easy': 2, 'moderate': 3,
                     'moderate to strenuous': 4, 'challenging': 5, 'very challenging': 6}
fitness_map       = {'basic': 1, 'moderate': 2, 'good': 3, 'excellent': 4}
avail_map         = {'very limited': 1, 'limited': 2, 'moderate': 3, 'good': 4, 'excellent': 5}
network_map       = {'none': 1, 'very limited': 2, 'limited': 3,
                     'moderate': 4, 'good': 5, 'excellent': 6}
risk_map          = {'very low': 1, 'low': 2, 'moderate': 3, 'high': 4, 'very high': 5}
altitude_risk_map = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3, 'very high': 4}


def _synthesize_seasonality(trek):
    """Generate season ratings (1-5) based on altitude and temperature."""
    alt = trek['max_altitude_m']
    temp_min = trek['temperature_min']
    temp_max = trek['temperature_max']

    spring, autumn, summer, winter = 4.0, 4.0, 3.0, 3.0

    if alt > 5000:
        winter = max(1.0, winter - 2.0)
        summer = max(1.0, summer - 1.0)
        spring = min(5.0, spring + 0.5)
        autumn = min(5.0, autumn + 1.0)
    elif alt > 4000:
        winter = max(1.0, winter - 1.0)
        autumn = min(5.0, autumn + 0.5)

    if temp_min < -10:
        winter = max(1.0, winter - 1.0)
    elif temp_min > 5:
        winter = min(5.0, winter + 1.0)
        summer = min(5.0, summer + 0.5)

    if alt < 3000 and temp_max > 20:
        winter = min(5.0, winter + 1.5)
        summer = max(1.0, summer - 0.5)

    noise = np.random.normal(0, 0.3, 4)
    ratings = np.clip([spring + noise[0], summer + noise[1],
                       autumn + noise[2], winter + noise[3]], 1.0, 5.0)
    return {
        'spring': round(float(ratings[0]), 1),
        'summer': round(float(ratings[1]), 1),
        'autumn': round(float(ratings[2]), 1),
        'winter': round(float(ratings[3]), 1),
    }


def _synthesize_altitude_sickness(trek):
    """Generate AMS details from existing risk and altitude data."""
    risk_str = trek['altitude_sickness_risk']
    alt = trek['max_altitude_m']

    if alt < 3500:
        acclim_days = 0
    elif alt < 4500:
        acclim_days = np.random.choice([1, 2])
    elif alt < 5500:
        acclim_days = np.random.choice([2, 3])
    else:
        acclim_days = np.random.choice([3, 4, 5])

    highest_pass = alt + np.random.randint(0, 300) if alt > 4000 else alt

    return {
        'risk_level': risk_str,
        'acclimatization_days': int(acclim_days),
        'highest_pass_m': int(highest_pass),
    }


def _synthesize_permits(trek):
    """Generate permit requirements based on existing permits_required flag."""
    needs_permit = trek['permits_required']
    alt = trek['max_altitude_m']
    difficulty_val = difficulty_map[trek['difficulty']]

    permit_types, total_cost, advance_days = [], 0, 0

    if needs_permit:
        permit_types.append('TIMS')
        total_cost += 20

        if trek.get('average_rating', 0) > 4.3 or alt > 4000:
            permit_types.append('national_park')
            total_cost += 30

        if difficulty_val >= 5 or alt > 5500:
            permit_types.append('restricted_area')
            total_cost += np.random.choice([50, 100, 150])
            advance_days = np.random.choice([7, 14, 21])

        if advance_days == 0:
            advance_days = np.random.choice([1, 2, 3])

        total_cost = int(total_cost * np.random.uniform(0.9, 1.2))

    return {
        'types': permit_types,
        'total_cost_usd': total_cost,
        'advance_booking_days': int(advance_days),
    }


def _synthesize_accommodation(trek):
    """Generate accommodation info based on trek characteristics."""
    alt = trek['max_altitude_m']
    difficulty_val = difficulty_map[trek['difficulty']]
    food_avail = avail_map[trek['food_availability']]

    types_available = []
    quality = 3

    if food_avail >= 3:
        types_available.append('teahouse')
        if food_avail >= 4:
            quality = min(5, quality + 1)
    if food_avail >= 4 and alt < 5000:
        types_available.append('lodge')
        quality = min(5, quality + 1)
    if difficulty_val >= 4 or food_avail <= 2:
        types_available.append('camping')
        quality = max(1, quality - 1)
    if alt < 4000 and difficulty_val <= 3:
        types_available.append('homestay')
    if not types_available:
        types_available.append('camping')
        quality = 2

    quality = int(np.clip(quality + np.random.choice([-1, 0, 0, 1]), 1, 5))

    return {'types': types_available, 'quality_rating': quality}


def _synthesize_transportation(trek):
    """Generate transport info based on remoteness."""
    alt = trek['max_altitude_m']
    difficulty_val = difficulty_map[trek['difficulty']]
    distance = trek['distance_km']

    if distance < 50:
        drive_hours = np.random.uniform(3, 6)
    elif distance < 100:
        drive_hours = np.random.uniform(5, 10)
    else:
        drive_hours = np.random.uniform(8, 16)

    flight_required = False
    if difficulty_val >= 5 and distance > 80:
        flight_required = True
        drive_hours = 0
    elif alt > 5000 and distance > 100:
        flight_required = np.random.random() < 0.6

    return {
        'drive_hours_from_ktm': round(float(drive_hours), 1),
        'flight_required': bool(flight_required),
    }


def _synthesize_health_safety(trek):
    """Generate health & safety info based on route characteristics."""
    alt = trek['max_altitude_m']
    nearest_medical = trek['nearest_medical_facility_km']
    evac = trek['evacuation_possible']

    if nearest_medical < 5:
        medical_posts = np.random.choice([2, 3, 4])
    elif nearest_medical < 15:
        medical_posts = np.random.choice([1, 2])
    else:
        medical_posts = np.random.choice([0, 1])

    heli_available = bool(evac)
    if alt > 6000 and not evac:
        heli_available = False

    if alt > 5000:
        oxygen_available = np.random.random() < 0.7
    elif alt > 4000:
        oxygen_available = np.random.random() < 0.3
    else:
        oxygen_available = False

    return {
        'medical_posts_on_route': int(medical_posts),
        'helicopter_evac': heli_available,
        'oxygen_available': bool(oxygen_available),
    }


# Apply synthesis — stored as metadata on each trek dict
for t in treks:
    t['seasonality'] = _synthesize_seasonality(t)
    t['altitude_sickness_detail'] = _synthesize_altitude_sickness(t)
    t['permit_details'] = _synthesize_permits(t)
    t['accommodation'] = _synthesize_accommodation(t)
    t['transportation'] = _synthesize_transportation(t)
    t['health_safety'] = _synthesize_health_safety(t)

# Build a fast lookup by trek_id
trek_by_id = {t['trek_id']: t for t in treks}

print(f"  Synthesized 6 enhanced attribute groups for {n_treks} treks")
print(f"  Sample trek seasonality  : {treks[0]['seasonality']}")
print(f"  Sample trek permits      : {treks[0]['permit_details']}")
print(f"  Sample trek accommodation: {treks[0]['accommodation']}")
print(f"  NOTE: These are stored as metadata only — NOT in the feature matrix")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 : Encode Trek Features (original 24 columns — unchanged)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 2 : Encoding Trek Features (24 columns)")
print("=" * 65)

trek_ids   = [t['trek_id'] for t in treks]
trek_names = {t['trek_id']: t['name'] for t in treks}

FEATURE_COLS = [
    'difficulty', 'fitness', 'risk', 'altitude_risk',
    'water', 'food', 'network',
    'permits', 'guide', 'evacuation',
    'duration_days', 'distance_km', 'max_altitude_m', 'altitude_gain_m',
    'daily_hours', 'cost_min', 'cost_max',
    'temp_min', 'temp_max', 'nearest_medical_km',
    'avg_rating', 'popularity', 'group_min', 'group_max'
]

trek_features = []
for t in treks:
    trek_features.append([
        difficulty_map[t['difficulty']],
        fitness_map[t['fitness_level_required']],
        risk_map[t['risk_level']],
        altitude_risk_map[t['altitude_sickness_risk']],
        avail_map[t['water_availability']],
        avail_map[t['food_availability']],
        network_map[t['mobile_network']],
        int(t['permits_required']),
        int(t['guide_mandatory']),
        int(t['evacuation_possible']),
        t['duration_days'],
        t['distance_km'],
        t['max_altitude_m'],
        t['altitude_gain_m'],
        t['daily_trek_hours'],
        t['estimated_cost_min_usd'],
        t['estimated_cost_max_usd'],
        t['temperature_min'],
        t['temperature_max'],
        t['nearest_medical_facility_km'],
        t['average_rating'],
        t['popularity_score'],
        t['group_size_min'],
        t['group_size_max'],
    ])

trek_feature_df = pd.DataFrame(trek_features, index=trek_ids, columns=FEATURE_COLS)

# Scale features to [0, 1]
scaler = MinMaxScaler()
trek_feature_scaled = scaler.fit_transform(trek_feature_df.values)
trek_feature_scaled_df = pd.DataFrame(trek_feature_scaled, index=trek_ids, columns=FEATURE_COLS)

# Feature importance weights based on variance
feature_variance = trek_feature_scaled.var(axis=0)
feat_imp_weights = 0.5 + (feature_variance / (feature_variance.max() + 1e-10))

print(f"  Feature matrix : {trek_feature_scaled_df.shape[0]} treks x {trek_feature_scaled_df.shape[1]} features")
print(f"  Columns: {FEATURE_COLS}")
print(f"  Top-3 most informative features: "
      f"{[FEATURE_COLS[i] for i in np.argsort(feature_variance)[::-1][:3]]}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 : Generate 600 Synthetic Users (15 profiles x 40) — Enhanced
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 3 : Generating 600 Synthetic Users (Enhanced Preferences)")
print("=" * 65)

SEASONS = ['spring', 'summer', 'autumn', 'winter']
ACCOMMODATION_TYPES = ['teahouse', 'camping', 'lodge', 'homestay']

USER_PROFILES = [
    # (name, diff, budget, dur, fit, region,
    #  preferred_season, ams_concern, permit_willing, accom_pref)
    ('Beginner Budget',        1, 300,  7,  1, None,             'autumn',  5, True,  'teahouse'),
    ('Casual Explorer',        2, 500,  10, 2, None,             'spring',  4, True,  'teahouse'),
    ('Weekend Trekker',        2, 400,  5,  2, 'Kathmandu Valley','autumn', 3, True,  'lodge'),
    ('Nature Lover',           3, 700,  12, 3, 'Annapurna',      'spring',  3, True,  'homestay'),
    ('Photography Enthusiast', 3, 800,  14, 3, None,             'autumn',  2, True,  'teahouse'),
    ('Cultural Trekker',       2, 600,  10, 2, 'Mustang',        'spring',  4, True,  'homestay'),
    ('Family Trekker',         2, 1000, 10, 2, None,             'autumn',  5, True,  'lodge'),
    ('Fitness Enthusiast',     4, 1000, 14, 3, 'Manaslu',        'spring',  2, True,  'teahouse'),
    ('Adventure Seeker',       5, 2000, 18, 4, 'Khumbu',         'autumn',  1, True,  'camping'),
    ('High Altitude Lover',    5, 2500, 21, 4, 'Khumbu',         'spring',  1, True,  'camping'),
    ('Budget Backpacker',      3, 400,  14, 3, None,             'summer',  3, False, 'teahouse'),
    ('Luxury Trekker',         3, 5000, 14, 3, 'Annapurna',      'autumn',  4, True,  'lodge'),
    ('Solo Trekker',           4, 1500, 16, 4, None,             'spring',  2, True,  'teahouse'),
    ('Remote Explorer',        5, 3000, 20, 4, 'Dolpo',          'autumn',  1, True,  'camping'),
    ('Himalayan Enthusiast',   6, 4000, 25, 4, 'Dhaulagiri',     'spring',  1, True,  'camping'),
]

N_USERS = 600
USERS_PER_PROFILE = 40
users = []

for profile_idx, (name, diff, budget, dur, fit, region,
                  season, ams, permit_w, accom) in enumerate(USER_PROFILES):
    for j in range(USERS_PER_PROFILE):
        uid = f"user_{profile_idx * USERS_PER_PROFILE + j:04d}"

        noise_diff   = np.random.choice([-1, 0, 1])
        noise_budget = np.random.uniform(0.75, 1.35)
        noise_dur    = np.random.choice([-2, -1, 0, 1, 2])
        noise_fit    = np.random.choice([-1, 0, 1])
        noise_ams    = np.random.choice([-1, 0, 0, 1])

        user_season = season if np.random.random() > 0.2 else np.random.choice(SEASONS)
        user_accom  = accom  if np.random.random() > 0.15 else np.random.choice(ACCOMMODATION_TYPES)

        users.append({
            'user_id':    uid,
            'profile':    name,
            'preferences': {
                'difficulty':               max(1, min(6, diff + noise_diff)),
                'budget_max':               int(budget * noise_budget),
                'duration_max':             max(1, min(30, dur + noise_dur)),
                'fitness':                  max(1, min(4, fit + noise_fit)),
                'region':                   region,
                'preferred_season':         user_season,
                'ams_concern_level':        max(1, min(5, ams + noise_ams)),
                'permit_willingness':       permit_w,
                'accommodation_preference': user_accom,
            }
        })

user_ids = [u['user_id'] for u in users]
user_index = {u['user_id']: i for i, u in enumerate(users)}
trek_index = {tid: i for i, tid in enumerate(trek_ids)}

print(f"  Generated {len(users)} users from {len(USER_PROFILES)} profiles")
print(f"  Preference fields: difficulty, budget_max, duration_max, fitness, region,")
print(f"    preferred_season, ams_concern_level, permit_willingness, accommodation_preference")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 : Generate Interactions (45% density) — Enhanced Scoring
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 4 : Generating User-Trek Interactions (Enhanced Scoring)")
print("=" * 65)


def compute_base_score(user, trek):
    """
    Compute base affinity score using weighted feature matching.

    The CORE scoring (difficulty, budget, duration, fitness, region, quality)
    uses the same weights as the original notebook to preserve the ranking
    signal that surfaces popular, well-known treks.

    The NEW parameters (season, AMS, permits, accommodation) act as
    SMALL MODIFIERS (total weight ~2.0 out of ~14.5) — enough to
    differentiate between similarly-ranked treks but not enough to
    override a strong core match.

    Weight summary:
      Core (unchanged):
        Difficulty  3.0  |  Budget  2.5  |  Duration 2.0
        Fitness     2.0  |  Region  1.5  |  Quality  1.0
      New (small modifiers):
        Season      0.8  |  AMS comfort  0.5
        Permits     0.3  |  Accommodation 0.4
    """
    prefs = user['preferences']
    score = 0.0
    total_weight = 0.0

    # ── CORE SCORING (preserved from original) ───────────────────────────

    # Difficulty match (weight 3.0)
    diff_gap = abs(prefs['difficulty'] - difficulty_map[trek['difficulty']])
    diff_score = max(0.0, 5.0 - diff_gap * 1.5)
    score += diff_score * 3.0
    total_weight += 3.0 * 5.0

    # Budget match (weight 2.5)
    cost_mid = (trek['estimated_cost_min_usd'] + trek['estimated_cost_max_usd']) / 2
    if prefs['budget_max'] >= cost_mid:
        budget_score = 5.0
    else:
        ratio = prefs['budget_max'] / max(cost_mid, 1)
        budget_score = max(0.0, ratio * 5.0)
    score += budget_score * 2.5
    total_weight += 2.5 * 5.0

    # Duration match (weight 2.0)
    if prefs['duration_max'] >= trek['duration_days']:
        dur_score = 5.0
    else:
        ratio = prefs['duration_max'] / max(trek['duration_days'], 1)
        dur_score = max(0.0, ratio * 5.0)
    score += dur_score * 2.0
    total_weight += 2.0 * 5.0

    # Fitness match (weight 2.0)
    fit_gap = abs(prefs['fitness'] - fitness_map[trek['fitness_level_required']])
    fit_score = max(0.0, 5.0 - fit_gap * 1.2)
    score += fit_score * 2.0
    total_weight += 2.0 * 5.0

    # Region preference (weight 1.5) — RESTORED from notebook
    region_pref = prefs.get('region')
    if region_pref and region_pref == trek.get('region'):
        region_score = 5.0
    elif region_pref is None:
        region_score = 3.0  # neutral
    else:
        region_score = 1.0
    score += region_score * 1.5
    total_weight += 1.5 * 5.0

    # Trek rating quality (weight 1.0, scaled 4.2-5.0 -> 0-5)
    rating_raw = trek['average_rating']
    rating_score = np.clip((rating_raw - 4.2) / (5.0 - 4.2) * 5.0, 0.0, 5.0)
    score += rating_score * 1.0
    total_weight += 1.0 * 5.0

    # ── NEW MODIFIERS (small weights — refine, don't override) ───────────

    # Seasonality match (weight 0.8)
    preferred_season = prefs.get('preferred_season', 'autumn')
    season_rating = trek['seasonality'].get(preferred_season, 3.0)
    score += season_rating * 0.8
    total_weight += 0.8 * 5.0

    # AMS concern match (weight 0.5)
    ams_concern = prefs.get('ams_concern_level', 3)
    ams_risk = altitude_risk_map[trek['altitude_sickness_detail']['risk_level']]
    if ams_concern >= 4 and ams_risk >= 3:
        ams_score = 1.0
    elif ams_concern <= 2 or ams_risk <= 1:
        ams_score = 5.0
    else:
        ams_score = 3.0
    score += ams_score * 0.5
    total_weight += 0.5 * 5.0

    # Permit willingness (weight 0.3)
    permit_types = trek['permit_details']['types']
    if not prefs.get('permit_willingness', True) and len(permit_types) > 1:
        permit_score = 1.5
    else:
        permit_score = 4.5
    score += permit_score * 0.3
    total_weight += 0.3 * 5.0

    # Accommodation match (weight 0.4)
    accom_pref = prefs.get('accommodation_preference', 'teahouse')
    accom_types = trek['accommodation']['types']
    if accom_pref in accom_types:
        accom_score = 5.0
    else:
        accom_score = 2.5
    score += accom_score * 0.4
    total_weight += 0.4 * 5.0

    return (score / total_weight) * 5.0  # Normalize to 0-5 scale


# Generate interactions for 45% of user-trek pairs
INTERACTION_DENSITY = 0.45
interaction_data = []

for user in users:
    for trek in treks:
        if np.random.random() > INTERACTION_DENSITY:
            continue

        base_score = compute_base_score(user, trek)

        views = int(np.clip(int(base_score * 2 + np.random.normal(0, 0.5)), 0, 10))
        booked = bool(base_score > 4.2 and np.random.random() < 0.3)
        favorites = bool(base_score > 3.5 and np.random.random() < 0.4)

        if base_score > 3.0:
            rating = round(base_score + np.random.normal(0, 0.2), 1)
            rating = float(np.clip(rating, 1.0, 5.0))
        else:
            rating = 0.0

        time_spent = int(np.clip(int(base_score * 100 + np.random.normal(0, 20)), 0, 600))

        interaction_data.append({
            'user_id':            user['user_id'],
            'trek_id':            trek['trek_id'],
            'views':              views,
            'booked':             booked,
            'favorites':          favorites,
            'rating':             rating,
            'time_spent_seconds': time_spent,
        })

interaction_df = pd.DataFrame(interaction_data)

# ── Compute raw weights ──────────────────────────────────────────────────
interaction_df['weight_raw'] = (
    interaction_df['views'] * 0.3 +
    interaction_df['booked'].astype(float) * 3.0 +
    interaction_df['favorites'].astype(float) * 1.5 +
    (interaction_df['rating'] / 5.0) * 2.0 +
    (interaction_df['time_spent_seconds'] / 600.0)
)

# ── Min-Max normalize weights to [0.5, 5.0] ─────────────────────────────
min_w = interaction_df['weight_raw'].min()
max_w = interaction_df['weight_raw'].max()
interaction_df['weight'] = 0.5 + (interaction_df['weight_raw'] - min_w) * (5.0 - 0.5) / (max_w - min_w)

interaction_df['weight'] += np.random.normal(0, 0.1, size=len(interaction_df))
interaction_df['weight'] = interaction_df['weight'].clip(0.5, 5.0)

density = len(interaction_df) / (N_USERS * n_treks) * 100
print(f"  {len(interaction_df):,} interactions generated")
print(f"  Density: {density:.1f}%")
print(f"  Weight range: [{interaction_df['weight'].min():.2f}, {interaction_df['weight'].max():.2f}]")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 : Build Interaction Matrix & Train/Test Split
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 5 : Train/Test Split")
print("=" * 65)

train_df, test_df = train_test_split(interaction_df, test_size=0.2, random_state=42)

rows = [user_index[r] for r in train_df['user_id']]
cols = [trek_index[r] for r in train_df['trek_id']]
vals = train_df['weight'].values

train_matrix = csr_matrix((vals, (rows, cols)), shape=(N_USERS, n_treks))

print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
print(f"  Matrix: {N_USERS} users x {n_treks} treks")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 : ALS Recommender (from scratch)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 6 : Training ALS Model")
print("=" * 65)


class ALSRecommender:
    """
    Confidence-weighted ALS (implicit ALS) — implemented from scratch.
    c_ui = 1 + confidence_scale * w_ui.
    Loss: sum_ui c_ui * (w_ui - u_u @ v_i)^2 + lambda * ||.||^2
    """

    def __init__(self, n_factors=50, n_iterations=30, lambda_reg=0.05,
                 confidence_scale=5.0):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.confidence_scale = confidence_scale
        self.U = None
        self.V = None
        self.training_rmse = []

    def fit(self, R, verbose=True):
        n_users, n_items = R.shape
        self.U = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.01, (n_items, self.n_factors))

        R_csr = csr_matrix(R)
        R_csc = csc_matrix(R)
        lam_I = self.lambda_reg * np.eye(self.n_factors)

        if verbose:
            print(f"  Training ALS: {n_users} users, {n_items} treks, "
                  f"{self.n_factors} factors, {self.n_iterations} iters, "
                  f"lambda={self.lambda_reg}, conf_scale={self.confidence_scale}")

        for iteration in range(1, self.n_iterations + 1):
            VtV = self.V.T @ self.V
            for u in range(n_users):
                row = R_csr.getrow(u)
                if row.nnz == 0:
                    continue
                item_idx = row.indices
                w = row.data
                conf = 1.0 + self.confidence_scale * w
                V_u = self.V[item_idx]
                A = VtV + V_u.T @ (V_u * (conf - 1)[:, np.newaxis]) + lam_I
                b = V_u.T @ (conf * w)
                self.U[u] = np.linalg.solve(A, b)

            UtU = self.U.T @ self.U
            for i in range(n_items):
                col = R_csc.getcol(i).tocsr()
                if col.nnz == 0:
                    continue
                user_idx = col.indices
                w = col.data
                conf = 1.0 + self.confidence_scale * w
                U_i = self.U[user_idx]
                A = UtU + U_i.T @ (U_i * (conf - 1)[:, np.newaxis]) + lam_I
                b = U_i.T @ (conf * w)
                self.V[i] = np.linalg.solve(A, b)

            if iteration % 5 == 0 or iteration == 1:
                nnz_r, nnz_c = R_csr.nonzero()
                preds   = np.array([self.U[r] @ self.V[ci]
                                    for r, ci in zip(nnz_r, nnz_c)])
                actuals = np.array([R_csr[r, ci]
                                    for r, ci in zip(nnz_r, nnz_c)])
                rmse = np.sqrt(np.mean((preds - actuals) ** 2))
                self.training_rmse.append((iteration, rmse))
                if verbose:
                    print(f"    Iteration {iteration:>3d}  |  RMSE: {rmse:.4f}")

    def predict(self, user_i, trek_j):
        return float(np.clip(self.U[user_i] @ self.V[trek_j], 0.5, 5.0))

    def predict_all(self, user_i):
        return np.clip(self.U[user_i] @ self.V.T, 0.5, 5.0)


# ── Hyperparameter grid search ──────────────────────────────────────────
print("  Running hyperparameter grid search...")

hp_train_df, hp_val_df = train_test_split(train_df, test_size=0.15, random_state=7)
hp_rows = [user_index[r] for r in hp_train_df['user_id']]
hp_cols = [trek_index[r] for r in hp_train_df['trek_id']]
hp_matrix = csr_matrix(
    (hp_train_df['weight'].values, (hp_rows, hp_cols)),
    shape=(N_USERS, n_treks)
)

PARAM_GRID = [
    {'n_factors': 25,  'n_iterations': 20, 'lambda_reg': 0.10, 'confidence_scale':  1.0},
    {'n_factors': 25,  'n_iterations': 20, 'lambda_reg': 0.10, 'confidence_scale':  5.0},
    {'n_factors': 50,  'n_iterations': 20, 'lambda_reg': 0.05, 'confidence_scale':  1.0},
    {'n_factors': 50,  'n_iterations': 20, 'lambda_reg': 0.05, 'confidence_scale':  5.0},
    {'n_factors': 50,  'n_iterations': 20, 'lambda_reg': 0.10, 'confidence_scale':  5.0},
    {'n_factors': 75,  'n_iterations': 20, 'lambda_reg': 0.05, 'confidence_scale':  1.0},
    {'n_factors': 75,  'n_iterations': 20, 'lambda_reg': 0.05, 'confidence_scale':  5.0},
    {'n_factors': 50,  'n_iterations': 20, 'lambda_reg': 0.05, 'confidence_scale': 10.0},
    {'n_factors': 50,  'n_iterations': 20, 'lambda_reg': 0.02, 'confidence_scale':  1.0},
    {'n_factors': 50,  'n_iterations': 20, 'lambda_reg': 0.02, 'confidence_scale':  5.0},
]

best_val_rmse = float('inf')
best_params = PARAM_GRID[0]

for params in PARAM_GRID:
    m = ALSRecommender(**params)
    m.fit(hp_matrix, verbose=False)
    vp, va = [], []
    for _, row in hp_val_df.iterrows():
        ui, ti = user_index[row['user_id']], trek_index[row['trek_id']]
        vp.append(float(np.clip(m.U[ui] @ m.V[ti], 0.5, 5.0)))
        va.append(row['weight'])
    val_rmse = np.sqrt(np.mean((np.array(vp) - np.array(va)) ** 2))
    print(f"    f={params['n_factors']:>3}, lam={params['lambda_reg']:.2f}, "
          f"cs={params['confidence_scale']:>5.1f}  |  val RMSE: {val_rmse:.4f}")
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params

print(f"  Best: {best_params}  (val RMSE: {best_val_rmse:.4f})")

# ── Final model ───────────────────────────────────────────────────────────
best_params_final = {**best_params, 'n_iterations': 30}
als_model = ALSRecommender(**best_params_final)
als_model.fit(train_matrix)

# ════════════════════════════════════════════════════════════════════════════
#  STEP 7 : Content-Based Filtering (manual cosine similarity)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 7 : Content-Based Filtering")
print("=" * 65)


def cosine_similarity_manual(A, B, weights=None):
    if weights is not None:
        A = A * weights
        B = B * weights
    dot = A @ B.T
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    norm_A = np.where(norm_A == 0, 1e-10, norm_A)
    norm_B = np.where(norm_B == 0, 1e-10, norm_B)
    return dot / (norm_A @ norm_B.T)


def build_user_profile_vector(user_id, source_df, trek_feature_scaled_df):
    user_interactions = source_df[source_df['user_id'] == user_id]
    if len(user_interactions) == 0:
        return None
    trek_indices = [trek_index[tid] for tid in user_interactions['trek_id']]
    weights = user_interactions['weight'].values
    feature_matrix = trek_feature_scaled_df.iloc[trek_indices].values
    weighted_sum = (feature_matrix.T * weights).T.sum(axis=0)
    return weighted_sum / weights.sum()


def preferences_to_vector(preferences):
    """Map new-user preferences to a feature vector for CBF.
    Uses the original 24-column feature space only."""
    vec = np.median(trek_feature_scaled, axis=0).copy()

    diff_idx = FEATURE_COLS.index('difficulty')
    vec[diff_idx] = (preferences['difficulty'] - 1) / 5.0

    fit_idx = FEATURE_COLS.index('fitness')
    vec[fit_idx] = (preferences['fitness'] - 1) / 3.0

    dur_idx = FEATURE_COLS.index('duration_days')
    dur_min = trek_feature_df['duration_days'].min()
    dur_max = trek_feature_df['duration_days'].max()
    vec[dur_idx] = np.clip((preferences['duration_max'] - dur_min) / (dur_max - dur_min), 0, 1)

    cost_min_idx = FEATURE_COLS.index('cost_min')
    cost_max_idx = FEATURE_COLS.index('cost_max')
    budget = preferences['budget_max']
    cmin_range = trek_feature_df['cost_min']
    cmax_range = trek_feature_df['cost_max']
    vec[cost_min_idx] = np.clip((budget - cmin_range.min()) / (cmin_range.max() - cmin_range.min()), 0, 1)
    vec[cost_max_idx] = np.clip((budget - cmax_range.min()) / (cmax_range.max() - cmax_range.min()), 0, 1)

    return vec


print("  Cosine similarity: manual numpy implementation")
print("  User profiles: weighted mean of interacted trek features (24-dim)")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 8 : Hybrid Recommendation System + Post-Ranking Re-scorer
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 8 : Building Hybrid Recommendation System")
print("=" * 65)

user_interacted = {}
for uid in user_ids:
    mask = train_df['user_id'] == uid
    user_interacted[uid] = set(train_df.loc[mask, 'trek_id'].values)

user_interacted_all = {}
for uid in user_ids:
    mask = interaction_df['user_id'] == uid
    user_interacted_all[uid] = set(interaction_df.loc[mask, 'trek_id'].values)

user_weight_sums = train_df.groupby('user_id')['weight'].sum().to_dict()


def normalize_scores(scores):
    smin, smax = scores.min(), scores.max()
    if smax - smin < 1e-10:
        return np.full_like(scores, 2.5)
    return (scores - smin) / (smax - smin) * 5.0


def compute_preference_rerank_bonus(prefs, trek):
    """
    Lightweight post-ranking bonus based on enhanced attributes.
    Returns a value in [-0.3, +0.3] — small enough to re-order
    similarly-scored treks but not override a strong ALS/CBF match.
    """
    bonus = 0.0

    # Season match
    preferred_season = prefs.get('preferred_season', 'autumn')
    season_rating = trek['seasonality'].get(preferred_season, 3.0)
    if season_rating >= 4.0:
        bonus += 0.10
    elif season_rating <= 2.0:
        bonus -= 0.05

    # AMS concern
    ams_concern = prefs.get('ams_concern_level', 3)
    ams_risk = altitude_risk_map[trek['altitude_sickness_detail']['risk_level']]
    if ams_concern >= 4 and ams_risk >= 3:
        bonus -= 0.10
    elif ams_concern <= 2 and ams_risk >= 3:
        bonus += 0.05

    # Accommodation match
    accom_pref = prefs.get('accommodation_preference', 'teahouse')
    if accom_pref in trek['accommodation']['types']:
        bonus += 0.05

    # Permit willingness
    if not prefs.get('permit_willingness', True):
        if 'restricted_area' in trek['permit_details']['types']:
            bonus -= 0.05

    # Safety bonus for concerned users
    if ams_concern >= 4:
        if trek['health_safety']['helicopter_evac']:
            bonus += 0.03
        if trek['health_safety']['medical_posts_on_route'] >= 2:
            bonus += 0.02

    return np.clip(bonus, -0.3, 0.3)


def recommend_als(user_id, top_n=10):
    ui = user_index[user_id]
    scores = als_model.predict_all(ui)
    for tid in user_interacted.get(user_id, set()):
        scores[trek_index[tid]] = -1.0
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(trek_ids[i], trek_names[trek_ids[i]], float(scores[i])) for i in top_indices]


def recommend_cbf(user_id, top_n=10):
    profile = build_user_profile_vector(user_id, train_df, trek_feature_scaled_df)
    if profile is None:
        return []
    sim = cosine_similarity_manual(
        profile.reshape(1, -1), trek_feature_scaled.copy(),
        weights=feat_imp_weights
    )[0]
    for tid in user_interacted.get(user_id, set()):
        sim[trek_index[tid]] = -1.0
    top_indices = np.argsort(sim)[::-1][:top_n]
    return [(trek_ids[i], trek_names[trek_ids[i]], float(sim[i])) for i in top_indices]


def recommend_hybrid(user_id, top_n=10):
    """
    Hybrid ALS + CBF with adaptive alpha, then a LIGHTWEIGHT
    post-ranking re-score using enhanced trek attributes.

    Pipeline:
      1. ALS scores (collaborative signal)
      2. CBF scores (content signal from 24 real features)
      3. Blend with adaptive alpha -> get ranked list
      4. Apply small preference bonus (+/-0.3 max) to re-rank
    """
    ui = user_index[user_id]

    als_scores = als_model.predict_all(ui)

    profile = build_user_profile_vector(user_id, train_df, trek_feature_scaled_df)
    if profile is not None:
        cbf_scores = cosine_similarity_manual(
            profile.reshape(1, -1), trek_feature_scaled.copy(),
            weights=feat_imp_weights
        )[0]
    else:
        cbf_scores = np.zeros(n_treks)

    als_norm = normalize_scores(als_scores)
    cbf_norm = normalize_scores(cbf_scores)

    n_seen = len(user_interacted.get(user_id, set()))
    total_w = user_weight_sums.get(user_id, 0.0)
    count_factor   = min(0.85, n_seen   / 20.0)
    quality_factor = min(0.85, total_w  / 30.0)
    alpha = np.clip(0.5 * count_factor + 0.5 * quality_factor, 0.10, 0.90)

    final_scores = alpha * als_norm + (1 - alpha) * cbf_norm

    for tid in user_interacted.get(user_id, set()):
        final_scores[trek_index[tid]] = -1.0

    # Post-ranking re-score with enhanced attributes
    prefs = users[user_index[user_id]]['preferences']
    for i in range(n_treks):
        if final_scores[i] > 0:
            trek = trek_by_id[trek_ids[i]]
            final_scores[i] += compute_preference_rerank_bonus(prefs, trek)

    top_indices = np.argsort(final_scores)[::-1][:top_n]
    return [(trek_ids[i], trek_names[trek_ids[i]], float(final_scores[i])) for i in top_indices]


def recommend_from_preferences(preferences, top_n=10):
    """
    Recommend for a new user based on preferences.
    Uses CBF (24-feature) + post-ranking with enhanced attributes.
    """
    vec = preferences_to_vector(preferences)
    sim = cosine_similarity_manual(
        vec.reshape(1, -1), trek_feature_scaled.copy(),
        weights=feat_imp_weights
    )[0]

    # Get wider candidate pool, then re-rank with preferences
    candidate_n = min(top_n * 3, n_treks)
    candidate_indices = np.argsort(sim)[::-1][:candidate_n]

    reranked = []
    for i in candidate_indices:
        trek = trek_by_id[trek_ids[i]]
        bonus = compute_preference_rerank_bonus(preferences, trek)
        reranked.append((i, float(sim[i]) + bonus))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return [(trek_ids[i], trek_names[trek_ids[i]], score)
            for i, score in reranked[:top_n]]


def recommend_for_new_user(preferences, top_n=10):
    """Alias for recommend_from_preferences."""
    return recommend_from_preferences(preferences, top_n)


print(f"  Hybrid model: ALS + CBF (24 features) + post-ranking re-scorer")
print(f"  Post-ranking bonus range: [-0.3, +0.3] (refines, doesn't override)")
print(f"  Methods: recommend_als, recommend_cbf, recommend_hybrid,")
print(f"           recommend_from_preferences, recommend_for_new_user")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 9 : Evaluation
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 9 : Model Evaluation")
print("=" * 65)

test_preds = []
test_actuals = []
for _, row in test_df.iterrows():
    ui = user_index[row['user_id']]
    ti = trek_index[row['trek_id']]
    pred = als_model.predict(ui, ti)
    test_preds.append(pred)
    test_actuals.append(row['weight'])

test_preds = np.array(test_preds)
test_actuals = np.array(test_actuals)

rmse = np.sqrt(np.mean((test_preds - test_actuals) ** 2))
mae = np.mean(np.abs(test_preds - test_actuals))

print(f"  Test RMSE : {rmse:.4f}")
print(f"  Test MAE  : {mae:.4f}")

if rmse < 0.4:
    interpretation = "Excellent"
elif rmse < 0.6:
    interpretation = "Good"
elif rmse < 0.8:
    interpretation = "Acceptable"
else:
    interpretation = "Needs tuning"
print(f"  Interpretation: {interpretation}")

# ── Ranking Metrics ──────────────────────────────────────────────────────
print("\n  --- Ranking Metrics ---")

EVAL_K = 10
median_weight = test_df['weight'].median()

user_relevant = {}
for uid in user_ids:
    ut = test_df[test_df['user_id'] == uid]
    relevant = set(ut[ut['weight'] > median_weight]['trek_id'].values)
    if relevant:
        user_relevant[uid] = relevant


def _dcg(scores, k):
    s = np.asarray(scores[:k], dtype=float)
    return float(np.sum(s / np.log2(np.arange(2, len(s) + 2)))) if len(s) else 0.0


def _ndcg(rec_ids, relevant, k):
    if not relevant:
        return 0.0
    gains = [1.0 if tid in relevant else 0.0 for tid in rec_ids[:k]]
    ideal = [1.0] * min(k, len(relevant))
    idcg  = _dcg(ideal, k)
    return _dcg(gains, k) / idcg if idcg > 0 else 0.0


ndcg_list, prec_list, recall_list = [], [], []
eval_uids = [u for u in user_ids if u in user_relevant][:200]


def _recommend_hybrid_eval(user_id, top_n=10):
    """Evaluation-only hybrid: excludes training interactions only."""
    ui = user_index[user_id]
    als_scores = als_model.predict_all(ui)
    profile = build_user_profile_vector(user_id, train_df, trek_feature_scaled_df)
    if profile is not None:
        cbf_scores = cosine_similarity_manual(
            profile.reshape(1, -1), trek_feature_scaled.copy(),
            weights=feat_imp_weights
        )[0]
    else:
        cbf_scores = np.zeros(n_treks)

    als_norm = normalize_scores(als_scores)
    cbf_norm = normalize_scores(cbf_scores)

    n_seen  = len(user_interacted.get(user_id, set()))
    total_w = user_weight_sums.get(user_id, 0.0)
    count_factor   = min(0.85, n_seen   / 20.0)
    quality_factor = min(0.85, total_w  / 30.0)
    alpha = np.clip(0.5 * count_factor + 0.5 * quality_factor, 0.10, 0.90)

    final_scores = alpha * als_norm + (1 - alpha) * cbf_norm

    # Post-ranking bonus
    prefs = users[user_index[user_id]]['preferences']
    for i in range(n_treks):
        if final_scores[i] > 0:
            trek = trek_by_id[trek_ids[i]]
            final_scores[i] += compute_preference_rerank_bonus(prefs, trek)

    for tid in user_interacted.get(user_id, set()):
        final_scores[trek_index[tid]] = -1.0
    top_idx = np.argsort(final_scores)[::-1][:top_n]
    return [trek_ids[i] for i in top_idx]


for uid in eval_uids:
    relevant = user_relevant[uid]
    rec_ids = _recommend_hybrid_eval(uid, top_n=EVAL_K)
    hits = len(set(rec_ids) & relevant)
    ndcg_list.append(_ndcg(rec_ids, relevant, EVAL_K))
    prec_list.append(hits / EVAL_K)
    recall_list.append(hits / len(relevant))

print(f"  NDCG@{EVAL_K}       : {np.mean(ndcg_list):.4f}")
print(f"  Precision@{EVAL_K}  : {np.mean(prec_list):.4f}")
print(f"  Recall@{EVAL_K}     : {np.mean(recall_list):.4f}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 10 : Recommendation Results
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 10 : Sample Recommendations")
print("=" * 65)

sample_users_list = ['user_0000', 'user_0001', 'user_0007', 'user_0009']
user_profiles_map = {u['user_id']: u['profile'] for u in users}

print("\n  --- Hybrid Recommendations (Existing Users) ---")
for uid in sample_users_list:
    profile = user_profiles_map[uid]
    user_prefs = users[user_index[uid]]['preferences']
    recs = recommend_hybrid(uid, top_n=5)
    print(f"\n  User: {uid}  |  Profile: {profile}")
    print(f"    Season: {user_prefs['preferred_season']}  |  "
          f"AMS concern: {user_prefs['ams_concern_level']}  |  "
          f"Accom: {user_prefs['accommodation_preference']}")
    print(f"  {'#':<4} {'Trek Name':<42} {'Score':>6}")
    print(f"  {'---':<4} {'-' * 42}  {'-' * 6}")
    for rank, (tid, name, score) in enumerate(recs, 1):
        trek = trek_by_id[tid]
        season_r = trek['seasonality'].get(user_prefs['preferred_season'], 0)
        accom_match = 'Y' if user_prefs['accommodation_preference'] in trek['accommodation']['types'] else 'N'
        ams_risk = trek['altitude_sickness_detail']['risk_level']
        stars = int(round(score))
        star_str = "*" * stars + "." * (5 - stars)
        print(f"  {rank:<4} {name[:42]:<42} {score:>6.3f}  {star_str}"
              f"  [S:{season_r:.0f} A:{accom_match} AMS:{ams_risk}]")

# New user recommendations with enhanced preferences
print("\n  --- New User Recommendations (Cold Start - Enhanced) ---")
new_user_scenarios = [
    ("Beginner, autumn, AMS-concerned, teahouse",
     {'difficulty': 2, 'budget_max': 400, 'duration_max': 7, 'fitness': 1,
      'preferred_season': 'autumn', 'ams_concern_level': 5,
      'permit_willingness': True, 'accommodation_preference': 'teahouse'}),

    ("Experienced, spring, AMS-unconcerned, camping",
     {'difficulty': 5, 'budget_max': 2500, 'duration_max': 18, 'fitness': 4,
      'preferred_season': 'spring', 'ams_concern_level': 1,
      'permit_willingness': True, 'accommodation_preference': 'camping'}),

    ("Moderate, winter, permit-averse, lodge",
     {'difficulty': 3, 'budget_max': 800, 'duration_max': 10, 'fitness': 3,
      'preferred_season': 'winter', 'ams_concern_level': 3,
      'permit_willingness': False, 'accommodation_preference': 'lodge'}),

    ("Family, summer, very concerned, homestay",
     {'difficulty': 2, 'budget_max': 1000, 'duration_max': 8, 'fitness': 2,
      'preferred_season': 'summer', 'ams_concern_level': 5,
      'permit_willingness': True, 'accommodation_preference': 'homestay'}),
]

for label, prefs in new_user_scenarios:
    recs = recommend_for_new_user(prefs, top_n=5)
    print(f"\n  Scenario: {label}")
    print(f"    Prefs: {prefs}")
    for rank, (tid, name, score) in enumerate(recs, 1):
        trek = trek_by_id[tid]
        season_r = trek['seasonality'].get(prefs.get('preferred_season', 'autumn'), 0)
        accom_match = 'Y' if prefs.get('accommodation_preference') in trek['accommodation']['types'] else 'N'
        print(f"    {rank}. {name[:42]:<42} (score: {score:.3f})"
              f"  [S:{season_r:.0f} A:{accom_match}]")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 11 : Save Model
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 11 : Saving Model")
print("=" * 65)

os.makedirs('models', exist_ok=True)

model_data = {
    'als_U':                als_model.U,
    'als_V':                als_model.V,
    'als_n_factors':        als_model.n_factors,
    'als_lambda_reg':       als_model.lambda_reg,
    'als_confidence_scale': als_model.confidence_scale,
    'als_training_rmse':    als_model.training_rmse,
    'best_hyperparams':     best_params_final,
    'trek_ids':             trek_ids,
    'trek_names':           trek_names,
    'trek_feature_scaled':  trek_feature_scaled,
    'trek_feature_df':      trek_feature_df,
    'feat_imp_weights':     feat_imp_weights,
    'scaler':               scaler,
    'feature_cols':         FEATURE_COLS,
    'user_ids':             user_ids,
    'users':                users,
    'user_index':           user_index,
    'trek_index':           trek_index,
    'interaction_df':       interaction_df,
    'train_df':             train_df,
    'user_interacted':      user_interacted,
    'user_weight_sums':     user_weight_sums,
    'test_rmse':            rmse,
    'test_mae':             mae,
    'ndcg_at_10':           float(np.mean(ndcg_list)) if ndcg_list else None,
    'precision_at_10':      float(np.mean(prec_list)) if prec_list else None,
    'recall_at_10':         float(np.mean(recall_list)) if recall_list else None,
    # Enhanced trek data for downstream use (API, frontend, etc.)
    'treks_enhanced':       treks,
    'trek_by_id':           trek_by_id,
    'seasons':              SEASONS,
    'accommodation_types':  ACCOMMODATION_TYPES,
}

with open('models/recommender.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"  Saved to models/recommender.pkl")
print(f"  Model includes enhanced trek metadata for downstream use")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 12 : Visualization
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  STEP 12 : Generating Visualizations")
print("=" * 65)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('ALS Matrix Factorization - Enhanced Hybrid Trek Recommendation Model',
             fontsize=14, fontweight='bold', y=0.98)

# Plot 1: Training RMSE per iteration
ax1 = axes[0, 0]
iters = [x[0] for x in als_model.training_rmse]
rmses = [x[1] for x in als_model.training_rmse]
ax1.plot(iters, rmses, 'o-', color='#2E75B6', linewidth=2, markersize=6)
ax1.axhline(y=rmse, color='red', linestyle='--', alpha=0.7, label=f'Test RMSE: {rmse:.4f}')
ax1.set_xlabel('ALS Iteration')
ax1.set_ylabel('RMSE')
ax1.set_title('Training RMSE per Iteration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual (test set)
ax2 = axes[0, 1]
sample_size = min(500, len(test_preds))
sample_idx = np.random.choice(len(test_preds), sample_size, replace=False)
ax2.scatter(test_actuals[sample_idx], test_preds[sample_idx],
            alpha=0.3, s=10, color='#375623')
ax2.plot([0, 5], [0, 5], 'r--', alpha=0.7, label='Perfect prediction')
ax2.set_xlabel('Actual Rating')
ax2.set_ylabel('Predicted Rating')
ax2.set_title('Predicted vs Actual Ratings (Test Set)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Interaction weight distribution
ax3 = axes[0, 2]
ax3.hist(interaction_df['weight'], bins=30, color='#4472C4', edgecolor='white', alpha=0.8)
ax3.set_xlabel('Weight')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Interaction Weights')
ax3.grid(True, alpha=0.3)

# Plot 4: Top 10 most recommended treks
ax4 = axes[1, 0]
rec_counts = {}
sample_rec_users = np.random.choice(user_ids, size=min(100, len(user_ids)), replace=False)
for uid in sample_rec_users:
    recs = recommend_hybrid(uid, top_n=5)
    for tid, name, score in recs:
        short_name = name[:25]
        rec_counts[short_name] = rec_counts.get(short_name, 0) + 1

top_10 = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:10]
if top_10:
    names_list = [x[0] for x in reversed(top_10)]
    counts_list = [x[1] for x in reversed(top_10)]
    ax4.barh(names_list, counts_list, color='#4472C4')
ax4.set_xlabel('Recommendation Count')
ax4.set_title('Top 10 Most Recommended Treks')

# Plot 5: Season distribution across treks (NEW)
ax5 = axes[1, 1]
season_data = {s: [] for s in SEASONS}
for t in treks:
    for s in SEASONS:
        season_data[s].append(t['seasonality'][s])
season_means = [np.mean(season_data[s]) for s in SEASONS]
season_colors = ['#66BB6A', '#FFA726', '#EF5350', '#42A5F5']
ax5.bar(SEASONS, season_means, color=season_colors, edgecolor='white')
ax5.set_ylabel('Average Season Rating (1-5)')
ax5.set_title('Trek Suitability by Season')
ax5.set_ylim(0, 5)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: AMS risk vs accommodation quality (NEW)
ax6 = axes[1, 2]
ams_risks = [altitude_risk_map[t['altitude_sickness_detail']['risk_level']] for t in treks]
accom_quals = [t['accommodation']['quality_rating'] for t in treks]
ax6.scatter(ams_risks, accom_quals, alpha=0.5, s=40, color='#AB47BC', edgecolor='white')
ax6.set_xlabel('Altitude Sickness Risk (0-4)')
ax6.set_ylabel('Accommodation Quality (1-5)')
ax6.set_title('AMS Risk vs Accommodation Quality')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('als_training_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved als_training_results.png")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  DONE - Enhanced Hybrid Trek Recommendation System Complete")
print("=" * 65)
print(f"  Users: {N_USERS} | Treks: {n_treks} | Interactions: {len(interaction_df):,}")
print(f"  Core features: {len(FEATURE_COLS)} (drives ALS + CBF)")
print(f"  Enhanced metadata: 6 attribute groups (drives post-ranking)")
print(f"  Best ALS params: {best_params_final}")
print(f"  Test RMSE: {rmse:.4f} ({interpretation})  |  MAE: {mae:.4f}")
if ndcg_list:
    print(f"  NDCG@10: {np.mean(ndcg_list):.4f}  |  "
          f"Precision@10: {np.mean(prec_list):.4f}  |  "
          f"Recall@10: {np.mean(recall_list):.4f}")
print(f"  Model saved: models/recommender.pkl")
print(f"  Plots saved: als_training_results.png")
