import requests

BASE_URL = "http://localhost:5001"


def ask_int(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Enter a valid integer.")


def ask_float(prompt, lo, hi):
    while True:
        try:
            v = float(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Enter a valid number.")


def collect_preferences():
    print("\n-- Preferences --")
    difficulty   = ask_int("Difficulty (1-6, where 1=easy 6=extreme): ", 1, 6)
    fitness      = ask_int("Fitness level (1-4, where 1=low 4=high): ", 1, 4)
    duration_max = ask_int("Max trip duration in days (1-30): ", 1, 30)
    budget_max   = ask_int("Max budget in USD (200-5000): ", 200, 5000)
    return {
        "difficulty":   difficulty,
        "fitness":      fitness,
        "duration_max": duration_max,
        "budget_max":   budget_max,
    }


def collect_interactions():
    interactions = []
    ans = input("\nDo you have interaction data to include? (y/n): ").strip().lower()
    if ans != 'y':
        return interactions

    print("\nEnter each trek interaction. Type 'done' as trek_id when finished.\n")
    while True:
        trek_id = input("Trek ID (or 'done'): ").strip()
        if trek_id.lower() == 'done':
            break
        if not trek_id:
            continue

        view_count = ask_int("  View count (0-100): ", 0, 100)
        booked     = input("  Booked? (y/n): ").strip().lower() == 'y'
        favorited  = input("  Favorited? (y/n): ").strip().lower() == 'y'
        rating_str = input("  Rating (1-5, or press Enter to skip): ").strip()
        rating     = float(rating_str) if rating_str else None
        time_spent = ask_int("  Time spent in seconds (0-3600): ", 0, 3600)

        interactions.append({
            "trek_id":            trek_id,
            "view_count":         view_count,
            "booked":             booked,
            "favorited":          favorited,
            "rating":             rating,
            "time_spent_seconds": time_spent,
        })
        print(f"  Added interaction for trek '{trek_id}'.\n")

    return interactions


def display_results(data):
    model   = data.get('model_used', '?')
    alpha   = data.get('alpha', 'N/A')
    n_inter = data.get('interaction_count', 0)

    print(f"\n{'='*60}")
    print(f"Model used : {model}")
    print(f"Alpha      : {alpha}  (ALS weight)")
    print(f"Interactions used: {n_inter}")
    print(f"{'='*60}\n")

    for i, r in enumerate(data['recommendations'], 1):
        print(f"{i:>2}. {r['name']}")
        print(f"    Score      : {r['score']}")
        print(f"    Difficulty : {r['difficulty']}  |  Fitness req : {r['fitness_required']}")
        print(f"    Duration   : {r['duration_days']} days  |  Max altitude : {r['max_altitude_m']} m")
        print(f"    Cost       : ${r['cost_min_usd']} - ${r['cost_max_usd']}")
        print(f"    Rating     : {r['avg_rating']}  |  Region : {r['region']}, {r['country']}")
        if r.get('short_description'):
            print(f"    {r['short_description']}")
        print()

    skipped = data.get('skipped_interactions', [])
    if skipped:
        print("Skipped interactions (unknown trek IDs):")
        for s in skipped:
            print(f"  trek_id={s['trek_id']}  reason={s['reason']}")


def main():
    print("=== Trek Recommender — CLI Test ===")

    prefs        = collect_preferences()
    interactions = collect_interactions()
    top_n        = ask_int("\nHow many recommendations to show? (1-20): ", 1, 20)

    payload = {
        "preferences":  prefs,
        "interactions": interactions,
        "top_n":        top_n,
    }

    print("\nFetching recommendations from API...")
    try:
        resp = requests.post(f"{BASE_URL}/recommend/hybrid", json=payload, timeout=10)
    except requests.ConnectionError:
        print(f"Could not connect to {BASE_URL}. Make sure the Flask server is running.")
        return

    if resp.status_code != 200:
        print(f"API error {resp.status_code}: {resp.json()}")
        return

    display_results(resp.json())


if __name__ == '__main__':
    main()
