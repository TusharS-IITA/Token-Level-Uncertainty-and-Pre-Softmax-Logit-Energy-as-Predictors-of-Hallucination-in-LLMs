import os
import re
import pandas as pd

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
}

def is_content_token(token_str: str) -> bool:
    clean_str = re.sub(r'[^\w\s]', '', token_str.strip().lower())
    if not clean_str:
        return False
    if clean_str in STOPWORDS:
        return False
    if len(clean_str) <= 1 and not clean_str.isdigit():
        return False
    return True

class TruthfulQALoader:
    def __init__(self, data_path: str = "data/TruthfulQA.csv"):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def get_evaluation_pairs(self, num_samples: int = 100):
        pairs = []
        for idx, row in self.df.iloc[:num_samples].iterrows():
            q = str(row['Question']).strip()
            best_ans = str(row['Best Answer']).strip()
            pairs.append({
                'id': f"q_{idx:03d}_true",
                'question': q,
                'answer': best_ans,
                'label': 0
            })
            inc_ans = str(row.get('Best Incorrect Answer', '')).strip()
            if not inc_ans or inc_ans == 'nan':
                raw_inc = str(row.get('Incorrect Answers', ''))
                inc_ans = raw_inc.split(';')[0].strip() if raw_inc else "I don't know"
            pairs.append({
                'id': f"q_{idx:03d}_false",
                'question': q,
                'answer': inc_ans,
                'label': 1
            })
        return pairs
