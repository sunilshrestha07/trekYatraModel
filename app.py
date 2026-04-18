import os
import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
np.random.seed(42)

# ── Load model ────────────────────────────────────────────────────────────────
with open('models/recommender.pkl', 'rb') as f:
    M = pickle.load(f)

trek_ids            = M['trek_ids']
trek_by_id          = M['trek_by_id']
trek_feature_scaled = M['trek_feature_scaled']
feat_imp_weights    = M['feat_imp_weights']
als_U               = M['als_U']
als_V               = M['als_V']
user_index          = M['user_index']
trek_index          = M['trek_index']
user_interacted     = M['user_interacted']
user_weight_sums    = M['user_weight_sums']
users               = M['users']
train_df            = M['train_df']
feature_cols        = M['feature_cols']
scaler              = M['scaler']
n_treks             = len(trek_ids)

raw_min = scaler.data_min_
raw_max = scaler.data_max_
col_idx = {c: i for i, c in enumerate(feature_cols)}

altitude_risk_map = {'none': 0, 'low': 1, 'moderate': 2, 'high': 3, 'very high': 4}

# ── Dynamic feature weighting for difficulty/fitness prioritisation ───────────

def dynamic_feature_weights():
    n_features = trek_feature_scaled.shape[1]
    other_weight = 0.4 / max(n_features - 2, 1)
    w = np.full(n_features, other_weight)
    w[col_idx['difficulty']] = 0.3
    w[col_idx['fitness']]    = 0.3
    return w


def difficulty_fitness_penalty(prefs):
    diff_idx = col_idx['difficulty']
    fit_idx  = col_idx['fitness']

    # Recover raw difficulty and fitness from scaled [0,1] values
    trek_diff_raw = (trek_feature_scaled[:, diff_idx]
                     * (raw_max[diff_idx] - raw_min[diff_idx]) + raw_min[diff_idx])
    trek_fit_raw  = (trek_feature_scaled[:, fit_idx]
                     * (raw_max[fit_idx] - raw_min[fit_idx]) + raw_min[fit_idx])

    user_diff = float(prefs['difficulty'])
    user_fit  = float(prefs['fitness'])

    penalty = np.zeros(n_treks)
    diff_excess = np.maximum(trek_diff_raw - user_diff, 0.0)
    fit_excess  = np.maximum(trek_fit_raw  - user_fit,  0.0)
    penalty = 0.25 * diff_excess + 0.25 * fit_excess

    return penalty


DYN_WEIGHTS = dynamic_feature_weights()

# Weight formula constants (must match train.py exactly)
W_VIEWS    = 0.3
W_BOOKED   = 3.0
W_FAVORITED = 1.5
W_RATING   = 2.0   # applied as (rating / 5.0) * W_RATING
W_TIME     = 1.0   # applied as (time_spent_seconds / 600.0) * W_TIME
W_MIN      = 0.5
W_MAX      = 5.0

print(f"  Loaded model: {len(trek_ids)} treks, {len(user_index)} users, "
      f"{als_U.shape[1]} ALS factors")


# ── Weight computation (mirrors train.py) ─────────────────────────────────────

def compute_weight(interaction: dict) -> float:
    view_count         = int(interaction.get('view_count', 0))
    booked             = bool(interaction.get('booked', False))
    favorited          = bool(interaction.get('favorited', False))
    rating             = interaction.get('rating')          # may be null
    time_spent_seconds = int(interaction.get('time_spent_seconds', 0))

    rating_term = (float(rating) / 5.0) * W_RATING if rating is not None else 0.0

    weight_raw = (
        view_count * W_VIEWS
        + float(booked) * W_BOOKED
        + float(favorited) * W_FAVORITED
        + rating_term
        + (min(time_spent_seconds, 600) / 600.0) * W_TIME
    )

    return float(np.clip(weight_raw, W_MIN, W_MAX))


# ── Interaction validation ────────────────────────────────────────────────────

def validate_interactions(raw_interactions: list) -> tuple[list, list]:
    valid   = []
    skipped = []

    for item in raw_interactions:
        tid = item.get('trek_id')

        # Must have a trek_id
        if not tid:
            skipped.append((None, 'missing trek_id'))
            continue

        # trek_id must exist in the model
        if tid not in trek_index:
            skipped.append((tid, 'unknown trek_id'))
            continue

        # Compute weight from raw signals
        weight = compute_weight(item)

        valid.append({
            'trek_id':            tid,
            'weight':             weight,
            'rating':             item.get('rating'),
            'view_count':         int(item.get('view_count', 0)),
            'booked':             bool(item.get('booked', False)),
            'favorited':          bool(item.get('favorited', False)),
            'time_spent_seconds': int(item.get('time_spent_seconds', 0)),
        })

    return valid, skipped


# ── Core math ─────────────────────────────────────────────────────────────────

def cosine_sim(A, B, weights=None):
    if weights is not None:
        A, B = A * weights, B * weights
    dot = A @ B.T
    nA = np.linalg.norm(A, axis=1, keepdims=True).clip(1e-10)
    nB = np.linalg.norm(B, axis=1, keepdims=True).clip(1e-10)
    return dot / (nA @ nB.T)


def normalize_scores(s):
    lo, hi = s.min(), s.max()
    return np.full_like(s, 2.5) if hi - lo < 1e-10 else (s - lo) / (hi - lo) * 5.0


def preferences_to_vector(prefs):
    vec = np.median(trek_feature_scaled, axis=0).copy()
    vec[col_idx['difficulty']]  = (prefs['difficulty'] - 1) / 5.0
    vec[col_idx['fitness']]     = (prefs['fitness'] - 1) / 3.0
    d_min, d_max = raw_min[col_idx['duration_days']], raw_max[col_idx['duration_days']]
    vec[col_idx['duration_days']] = np.clip(
        (prefs['duration_max'] - d_min) / (d_max - d_min + 1e-10), 0, 1)
    budget = prefs['budget_max']
    for col in ('cost_min', 'cost_max'):
        lo, hi = raw_min[col_idx[col]], raw_max[col_idx[col]]
        vec[col_idx[col]] = np.clip((budget - lo) / (hi - lo + 1e-10), 0, 1)
    return vec


def user_profile_vector(user_id):
    """Build CBF profile vector from training interactions (legacy users)."""
    rows = train_df[train_df['user_id'] == user_id]
    if len(rows) == 0:
        return None
    idxs = [trek_index[tid] for tid in rows['trek_id']]
    w    = rows['weight'].values
    return (trek_feature_scaled[idxs].T * w).T.sum(axis=0) / w.sum()


def rerank_bonus(prefs, trek):
    bonus = 0.0
    season = prefs.get('preferred_season', 'autumn')
    sr = trek.get('seasonality', {}).get(season, 3.0)
    if sr >= 4.0:   bonus += 0.10
    elif sr <= 2.0: bonus -= 0.05

    ams  = prefs.get('ams_concern_level', 3)
    risk = altitude_risk_map.get(
        trek.get('altitude_sickness_detail', {}).get('risk_level', 'moderate'), 2)
    if ams >= 4 and risk >= 3:   bonus -= 0.10
    elif ams <= 2 and risk >= 3: bonus += 0.05

    if prefs.get('accommodation_preference', 'teahouse') in \
            trek.get('accommodation', {}).get('types', []):
        bonus += 0.05

    if not prefs.get('permit_willingness', True):
        if 'restricted_area' in trek.get('permit_details', {}).get('types', []):
            bonus -= 0.05

    if ams >= 4:
        hs = trek.get('health_safety', {})
        if hs.get('helicopter_evac'):                  bonus += 0.03
        if hs.get('medical_posts_on_route', 0) >= 2:  bonus += 0.02

    return float(np.clip(bonus, -0.3, 0.3))


def fmt(tid, score):
    t = trek_by_id[tid]
    return {
        'trek_id':          tid,
        'name':             t['name'],
        'score':            round(float(score), 4),
        'difficulty':       t['difficulty'],
        'duration_days':    t['duration_days'],
        'distance_km':      t['distance_km'],
        'max_altitude_m':   t['max_altitude_m'],
        'cost_min_usd':     t['estimated_cost_min_usd'],
        'cost_max_usd':     t['estimated_cost_max_usd'],
        'fitness_required': t['fitness_level_required'],
        'avg_rating':       t['average_rating'],
        'region':           t['region'],
        'country':          t['country'],
        'short_description': t.get('short_description', ''),
    }


# ── Recommendation engines ────────────────────────────────────────────────────

def recommend_cbf(prefs, top_n=10):
    """Pure content-based filtering. Used for cold-start (no interactions)."""
    sim  = cosine_sim(preferences_to_vector(prefs).reshape(1, -1),
                      trek_feature_scaled, weights=DYN_WEIGHTS)[0]
    pen  = difficulty_fitness_penalty(prefs)
    scores = sim - pen
    pool = np.argsort(scores)[::-1][:top_n * 3]
    ranked = sorted(
        [(i, float(scores[i]) + rerank_bonus(prefs, trek_by_id[trek_ids[i]]))
         for i in pool],
        key=lambda x: -x[1]
    )
    return [fmt(trek_ids[i], s) for i, s in ranked[:top_n]]


def recommend_hybrid_runtime(prefs, valid_interactions, top_n=10):
    interacted_ids = {ia['trek_id'] for ia in valid_interactions}

    # ── CBF score (from preferences) ────────────────────────────────────────
    vec   = preferences_to_vector(prefs)
    cbf_s = cosine_sim(vec.reshape(1, -1), trek_feature_scaled,
                       weights=DYN_WEIGHTS)[0]

    # ── ALS fold-in (from interactions) ─────────────────────────────────────
    weighted_vecs = []
    weights       = []
    for ia in valid_interactions:
        idx = trek_index[ia['trek_id']]
        w   = ia['weight']
        weighted_vecs.append(als_V[idx] * w)
        weights.append(w)

    total_weight = sum(weights)
    user_vec     = np.sum(weighted_vecs, axis=0) / total_weight
    als_s        = np.clip(user_vec @ als_V.T, W_MIN, W_MAX)

    # ── Adaptive alpha ───────────────────────────────────────────────────────
    n_seen = len(valid_interactions)
    alpha  = float(np.clip(
        0.5 * min(0.85, n_seen / 20.0) + 0.5 * min(0.85, total_weight / 30.0),
        0.10, 0.90
    ))

    # ── Blend ────────────────────────────────────────────────────────────────
    final = alpha * normalize_scores(als_s) + (1 - alpha) * normalize_scores(cbf_s)

    # Apply difficulty/fitness soft penalty after blending
    final -= difficulty_fitness_penalty(prefs)

    # Exclude already-interacted treks
    for tid in interacted_ids:
        if tid in trek_index:
            final[trek_index[tid]] = -1.0

    # Rerank bonus
    for i in range(n_treks):
        if final[i] > 0:
            final[i] += rerank_bonus(prefs, trek_by_id[trek_ids[i]])

    top = np.argsort(final)[::-1][:top_n]
    return [fmt(trek_ids[i], final[i]) for i in top], alpha


def recommend_als_user(user_id, top_n=10):
    """ALS-only for training users (legacy)."""
    scores = np.clip(als_U[user_index[user_id]] @ als_V.T, W_MIN, W_MAX)
    for tid in user_interacted.get(user_id, set()):
        if tid in trek_index:
            scores[trek_index[tid]] = -1.0
    top = np.argsort(scores)[::-1][:top_n]
    return [fmt(trek_ids[i], scores[i]) for i in top]


def recommend_hybrid_user(user_id, top_n=10):
    """Hybrid for training users — uses pre-trained U matrix (legacy)."""
    ui    = user_index[user_id]
    als_s = np.clip(als_U[ui] @ als_V.T, W_MIN, W_MAX)

    profile = user_profile_vector(user_id)
    cbf_s   = (cosine_sim(profile.reshape(1, -1), trek_feature_scaled,
                           weights=DYN_WEIGHTS)[0]
               if profile is not None else np.zeros(n_treks))

    interacted = user_interacted.get(user_id, set())
    n_seen     = len(interacted)
    total_w    = user_weight_sums.get(user_id, 0.0)
    alpha      = float(np.clip(
        0.5 * min(0.85, n_seen / 20.0) + 0.5 * min(0.85, total_w / 30.0),
        0.10, 0.90
    ))

    final = alpha * normalize_scores(als_s) + (1 - alpha) * normalize_scores(cbf_s)

    prefs = users[ui]['preferences']

    # Apply difficulty/fitness soft penalty after blending
    final -= difficulty_fitness_penalty(prefs)

    for tid in interacted:
        if tid in trek_index:
            final[trek_index[tid]] = -1.0

    for i in range(n_treks):
        if final[i] > 0:
            final[i] += rerank_bonus(prefs, trek_by_id[trek_ids[i]])

    top = np.argsort(final)[::-1][:top_n]
    return [fmt(trek_ids[i], final[i]) for i in top], alpha


# ── Validation helpers ────────────────────────────────────────────────────────

def _parse_prefs(prefs_dict: dict):
    """Validate and coerce preferences dict. Returns (prefs, top_n) or error."""
    required = ['difficulty', 'budget_max', 'duration_max', 'fitness']
    missing  = [f for f in required if f not in prefs_dict]
    if missing:
        return None, ({'error': f'Missing preference fields: {missing}'}, 400)

    errs = []
    try:
        if not 1 <= int(prefs_dict['difficulty']) <= 6:
            errs.append('difficulty must be 1-6')
        if not 1 <= int(prefs_dict['fitness']) <= 4:
            errs.append('fitness must be 1-4')
        if not 1 <= int(prefs_dict['duration_max']) <= 30:
            errs.append('duration_max must be 1-30')
        if not 200 <= int(prefs_dict['budget_max']) <= 5000:
            errs.append('budget_max must be 200-5000')
    except (ValueError, TypeError) as e:
        return None, ({'error': f'Invalid preference value: {e}'}, 400)

    if errs:
        return None, ({'error': errs}, 400)

    prefs = {k: int(prefs_dict[k]) for k in required}
    for k in ('preferred_season', 'ams_concern_level',
              'permit_willingness', 'accommodation_preference'):
        if k in prefs_dict:
            prefs[k] = prefs_dict[k]

    return prefs, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'JSON body required'}), 400

    # ── Parse preferences ────────────────────────────────────────────────────
    raw_prefs = body.get('preferences')
    if not raw_prefs or not isinstance(raw_prefs, dict):
        return jsonify({'error': '`preferences` object is required'}), 400

    prefs, err = _parse_prefs(raw_prefs)
    if err:
        return jsonify(err[0]), err[1]

    top_n = max(1, min(int(body.get('top_n', 10)), n_treks))

    # ── Validate interactions ────────────────────────────────────────────────
    raw_interactions = body.get('interactions', [])
    if not isinstance(raw_interactions, list):
        return jsonify({'error': '`interactions` must be an array'}), 400

    valid_interactions, skipped = validate_interactions(raw_interactions)

    # ── Route to correct engine ──────────────────────────────────────────────
    if not valid_interactions:
        # Cold start — pure CBF
        recs = recommend_cbf(prefs, top_n)
        response = {
            'recommendations': recs,
            'count':           len(recs),
            'model_used':      'cbf',
            'alpha':           0.0,
            'interaction_count': 0,
        }
    else:
        # Hybrid runtime
        recs, alpha = recommend_hybrid_runtime(prefs, valid_interactions, top_n)
        response = {
            'recommendations':   recs,
            'count':             len(recs),
            'model_used':        'hybrid_runtime',
            'alpha':             round(alpha, 4),
            'interaction_count': len(valid_interactions),
        }

    # Debug info (useful for understanding behaviour)
    if skipped:
        response['skipped_interactions'] = [
            {'trek_id': s[0], 'reason': s[1]} for s in skipped
        ]

    return jsonify(response)


@app.route('/recommend', methods=['POST'])
def recommend():
    """Pure CBF — no interactions needed."""
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'JSON required'}), 400

    raw_prefs = body.get('preferences', body)  # accept flat body too
    prefs, err = _parse_prefs(raw_prefs)
    if err:
        return jsonify(err[0]), err[1]

    top_n = max(1, min(int(body.get('top_n', 10)), n_treks))
    recs  = recommend_cbf(prefs, top_n)
    return jsonify({'recommendations': recs, 'count': len(recs), 'model_used': 'cbf'})


@app.route('/recommend/als', methods=['POST'])
def recommend_als():
    """ALS for training users; CBF fallback."""
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'JSON required'}), 400

    uid = body.get('user_id')
    if uid and uid in user_index:
        top_n = max(1, min(int(body.get('top_n', 10)), n_treks))
        recs  = recommend_als_user(uid, top_n)
        return jsonify({'recommendations': recs, 'count': len(recs),
                        'model_used': 'als', 'user_id': uid})
    if uid:
        return jsonify({'error': f'Unknown user_id: {uid}'}), 404

    raw_prefs = body.get('preferences', body)
    prefs, err = _parse_prefs(raw_prefs)
    if err:
        return jsonify(err[0]), err[1]

    top_n = max(1, min(int(body.get('top_n', 10)), n_treks))
    recs  = recommend_cbf(prefs, top_n)
    return jsonify({'recommendations': recs, 'count': len(recs), 'model_used': 'cbf_fallback'})


@app.route('/recommend/user/<user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """Full hybrid for a training user (legacy / testing)."""
    if user_id not in user_index:
        return jsonify({'error': f'Unknown user_id: {user_id}'}), 404

    top_n      = max(1, min(int(request.args.get('top_n', 10)), n_treks))
    recs, alpha = recommend_hybrid_user(user_id, top_n)
    ui         = user_index[user_id]
    prefs      = {
        k: (int(v)   if isinstance(v, np.integer)  else
            float(v) if isinstance(v, np.floating) else v)
        for k, v in users[ui]['preferences'].items()
    }
    return jsonify({
        'recommendations':        recs,
        'count':                  len(recs),
        'model_used':             'hybrid',
        'user_id':                user_id,
        'alpha':                  round(alpha, 4),
        'user_preferences':       prefs,
        'interaction_count':      len(user_interacted.get(user_id, set())),
        'total_interaction_weight': round(float(user_weight_sums.get(user_id, 0.0)), 2),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'treks_loaded': n_treks,
        'model_loaded': True,
        'known_users':  len(user_index),
        'als_factors':  int(als_U.shape[1]),
        'test_rmse':    M.get('test_rmse'),
        'test_mae':     M.get('test_mae'),
    })


@app.route('/treks/count', methods=['GET'])
def trek_count():
    return jsonify({'total_treks': n_treks})


@app.route('/users', methods=['GET'])
def list_users():
    ids = M.get('user_ids', [])
    return jsonify({'users': ids, 'count': len(ids)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
