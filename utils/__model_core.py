import numpy as np

LABELS = ['Religion', 'Age', 'Ethnicity', 'Gender', 'Not Bullying']

# ── Keyword maps ──────────────────────────────────────────────────────────────
KEYWORD_MAP = {
    'Religion': [
        'muslim', 'islam', 'islamic', 'christian', 'hindu', 'jewish', 'jew', 'sikh',
        'allah', 'god', 'bible', 'quran', 'religion', 'church', 'mosque', 'temple',
        'kafir', 'infidel', 'crusade', 'jihad', 'blasphemy', 'atheist', 'pagan',
        'buddhist', 'catholic', 'protestant', 'scripture', 'prayer', 'worship',
        'cult', 'sect', 'heretic', 'apostate', 'convert', 'baptism', 'heathen',
    ],
    'Age': [
        'boomer', 'old man', 'old woman', 'old people', 'elderly', 'ancient',
        'dinosaur', 'kid', 'immature', 'childish', 'too young', 'too old',
        'retire', 'senile', 'wrinkled', 'grandpa', 'grandma', 'granny',
        'millennial', 'gen z', 'zoomer', 'young people', 'children nowadays',
        'back in my day', 'ok boomer', 'these kids', 'old folk', 'over the hill',
        'past your prime', 'age is just', 'act your age', 'grow up',
    ],
    'Ethnicity': [
        'nigger', 'nigga', 'negro', 'black people', 'white people', 'chink',
        'gook', 'spic', 'wetback', 'beaner', 'cracker', 'honky', 'racist',
        'racism', 'racial', 'ethnic', 'immigrant', 'foreigner', 'go back to',
        'your country', 'skin color', 'colored', 'minority', 'white supremacy',
        'kkk', 'nazi', 'aryan', 'segregation', 'deportation', 'illegal alien',
        'anchor baby', 'curry', 'brownie', 'yellowface', 'blackface',
    ],
    'Gender': [
        'bitch', 'slut', 'whore', 'cunt', 'feminazi', 'women are', 'females are',
        'man up', 'like a girl', 'gay', 'lesbian', 'faggot', 'fag', 'tranny',
        'transgender', 'queer', 'sexist', 'misogyn', 'patriarchy', 'go back to kitchen',
        'mansplain', 'simp', 'incel', 'alpha male', 'beta male', 'women belong',
        'girls cant', 'boys dont cry', 'typical woman', 'typical man',
        'women driver', 'stay in kitchen', 'make me a sandwich', 'lgbtq',
        'nonbinary', 'females', 'enby',
    ],
}

INSULT_WORDS = {
    'stupid', 'idiot', 'moron', 'dumb', 'ugly', 'hate', 'disgusting', 'pathetic',
    'loser', 'trash', 'garbage', 'worthless', 'useless', 'freak', 'weirdo',
    'creep', 'terrible', 'horrible', 'awful', 'worst', 'retard', 'psycho',
    'degenerate', 'filth', 'scum', 'pig', 'animal', 'monster', 'sicko',
    'pervert', 'predator', 'lowlife', 'subhuman', 'vermin', 'parasite',
}

HARD_SLURS = {
    'nigger', 'nigga', 'faggot', 'fag', 'chink', 'spic', 'wetback',
    'beaner', 'gook', 'tranny', 'cunt', 'slut', 'whore', 'bitch',
    'retard', 'kafir', 'kike', 'cracker',
}





# ── Keyword classifier (offline) ──────────────────────────────────────────────
def _keyword_predict(text: str):
    """Pure keyword-based classifier. Instant, no dependencies."""
    lower = text.lower()

    # Step 1: hard slur — immediate high confidence
    for slur in HARD_SLURS:
        if slur in lower:
            for label, keywords in KEYWORD_MAP.items():
                if slur in keywords:
                    probs = np.full(5, 0.02, dtype=np.float32)
                    probs[LABELS.index(label)] = 0.92
                    probs /= probs.sum()
                    return label, probs
            probs = np.array([0.05, 0.02, 0.88, 0.03, 0.02], dtype=np.float32)
            return 'Ethnicity', probs

    # Step 2: category keyword scoring
    scores = {label: 0 for label in LABELS}
    for label, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in lower:
                scores[label] += 1

    total_matches = sum(scores.values())
    has_insult    = any(w in lower for w in INSULT_WORDS)
    best_label    = max(scores, key=scores.get)
    best_score    = scores[best_label]

    if best_score > 0:
        confidence = min(0.55 + best_score * 0.12, 0.93)
        if has_insult:
            confidence = min(confidence + 0.08, 0.95)
        probs = np.full(5, (1.0 - confidence) / 4, dtype=np.float32)
        probs[LABELS.index(best_label)] = confidence
        if total_matches > best_score:
            for i, label in enumerate(LABELS):
                if label != best_label and scores[label] > 0:
                    probs[i] += (scores[label] / total_matches) * 0.05
        probs /= probs.sum()
        return best_label, probs

    # Step 3: generic insult, no category
    if has_insult:
        probs = np.array([0.18, 0.12, 0.35, 0.20, 0.15], dtype=np.float32)
        return 'Ethnicity', probs

    # Step 4: clean text
    probs = np.array([0.03, 0.03, 0.03, 0.03, 0.88], dtype=np.float32)
    return 'Not Bullying', probs

# ── Main entry point ──────────────────────────────────────────────────────────
def predict_bilstm(text, model=None, vocab_to_int=None, max_len=None):
    return _keyword_predict(text)