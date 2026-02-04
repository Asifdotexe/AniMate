import json
import numpy as np # Used in the string, but helpful to import here to check syntax if needed
from pathlib import Path

def refine_notebook():
    nb_path = Path("notebooks/issue-58-as-poc-history-recommendation.ipynb")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        
        # 1. Add Documentation for Ranking Assumption
        # Insert into Cell 0 source
        first_cell_source = nb["cells"][0]["source"]
        if not any("## Assumption: Popularity & Quality Bias" in line for line in first_cell_source):
             # Append to the end of assumptions
             # Find where the last assumption ends or just append to end of cell
             new_doc = [
                 "\n",
                 "## Assumption: Popularity & Quality Bias\n",
                 "> **Assumption**: Users prefer \"good\" and \"popular\" shows over obscure ones, all else being equal. \n",
                 "> **Refinement**: After finding similar animes (by distance), we re-rank the candidates using a **Hybrid Score**:\n",
                 "> - **Similarity** (Distance)\n",
                 "> - **Popularity** (Favorites count, log-scaled)\n",
                 "> - **Quality** (MyAnimeList Score)\n",
                 "> This ensures we recommend high-confidence hits.\n"
             ]
             first_cell_source.extend(new_doc)

        # 2. Define New Function Code strings
        
        new_centroid_code = [
            "def recommend_centroid(history_titles, df, vectorizer, knn, top_k=5):\n",
            "    vectors, found_titles = get_user_history_vectors(history_titles, df, vectorizer)\n",
            "    \n",
            "    if vectors is None:\n",
            "        return pd.DataFrame()\n",
            "    \n",
            "    # Calculate Centroid\n",
            "    user_vector = vectors.mean(axis=0)\n",
            "    import numpy as np\n",
            "    user_vector = np.asarray(user_vector)\n",
            "    \n",
            "    # 1. Fetch Larger Pool (50) to allow filtering and re-ranking\n",
            "    distances, indices = knn.kneighbors(user_vector, n_neighbors=50)\n",
            "    \n",
            "    candidates = []\n",
            "    for dist, idx in zip(distances[0], indices[0]):\n",
            "        row = df.iloc[idx]\n",
            "        title = row['title']\n",
            "        \n",
            "        # Franchise Filtering\n",
            "        is_franchise_duplicate = any(history_item.lower() in title.lower() for history_item in found_titles)\n",
            "        if title in found_titles or is_franchise_duplicate:\n",
            "            continue\n",
            "            \n",
            "        candidates.append({\n",
            "            'title': title,\n",
            "            'genres': row.get('genres'),\n",
            "            'themes': row.get('themes'),\n",
            "            'distance': dist,\n",
            "            'favorites': row.get('favorites', 0),\n",
            "            'score': row.get('score', 0),\n",
            "            'strategy': 'Centroid'\n",
            "        })\n",
            "    \n",
            "    if not candidates:\n",
            "        return pd.DataFrame()\n",
            "        \n",
            "    # 2. Hybrid Scoring & Re-ranking\n",
            "    df_cand = pd.DataFrame(candidates)\n",
            "    \n",
            "    # Normalize Metrics (0-1 scale)\n",
            "    # Distance: Lower is better. Invert it. \n",
            "    # We use 1 - dist as a base similarity score (Clip at 0 for safety, though cosine dist is 0-1 usually)\n",
            "    df_cand['sim_score'] = 1 - df_cand['distance'].clip(0, 1)\n",
            "    \n",
            "    # Favorites: Log scale to handle power law (100 vs 100k)\n",
            "    df_cand['fav_log'] = np.log1p(df_cand['favorites'].fillna(0))\n",
            "    df_cand['fav_norm'] = df_cand['fav_log'] / (df_cand['fav_log'].max() + 1e-6)\n",
            "    \n",
            "    # Score: Already 0-10, normalize to 0-1\n",
            "    df_cand['score_norm'] = df_cand['score'].fillna(0) / 10.0\n",
            "    \n",
            "    # Weighted Combination\n",
            "    # 50% Similarity, 30% Popularity, 20% Quality\n",
            "    df_cand['final_score'] = (\n",
            "        0.5 * df_cand['sim_score'] + \n",
            "        0.3 * df_cand['fav_norm'] + \n",
            "        0.2 * df_cand['score_norm']\n",
            "    )\n",
            "    \n",
            "    # Sort and take top K\n",
            "    return df_cand.sort_values('final_score', ascending=False).head(top_k)\n"
        ]

        new_multiquery_code = [
            "def recommend_multiquery(history_titles, df, vectorizer, knn, top_k=5):\n",
            "    vectors, found_titles = get_user_history_vectors(history_titles, df, vectorizer)\n",
            "    \n",
            "    if vectors is None:\n",
            "        return pd.DataFrame()\n",
            "    \n",
            "    candidates = {}\n",
            "    \n",
            "    for i in range(vectors.shape[0]):\n",
            "        vec = vectors.getrow(i)\n",
            "        # Fetch larger pool per item\n",
            "        dists, idxs = knn.kneighbors(vec, n_neighbors=20)\n",
            "        \n",
            "        for d, idx in zip(dists[0], idxs[0]):\n",
            "            row = df.iloc[idx]\n",
            "            title = row['title']\n",
            "            \n",
            "            # Franchise Filtering\n",
            "            is_franchise_duplicate = any(history_item.lower() in title.lower() for history_item in found_titles)\n",
            "            if title in found_titles or is_franchise_duplicate:\n",
            "                continue\n",
            "                \n",
            "            # Base Similarity Score\n",
            "            score_boost = 1.0 - d\n",
            "            \n",
            "            if title not in candidates:\n",
            "                candidates[title] = {\n",
            "                    'row': row, \n",
            "                    'sim_sum': 0, \n",
            "                    'frequency': 0,\n",
            "                    'min_distance': 1.0\n",
            "                }\n",
            "            \n",
            "            candidates[title]['sim_sum'] += score_boost\n",
            "            candidates[title]['frequency'] += 1\n",
            "            candidates[title]['min_distance'] = min(candidates[title]['min_distance'], d)\n",
            "            \n",
            "    if not candidates:\n",
            "        return pd.DataFrame()\n",
            "        \n",
            "    results = []\n",
            "    for title, data in candidates.items():\n",
            "        results.append({\n",
            "            'title': title,\n",
            "            'genres': data['row'].get('genres'),\n",
            "            'themes': data['row'].get('themes'),\n",
            "            'sim_sum': data['sim_sum'],\n",
            "            'frequency': data['frequency'],\n",
            "            'best_dist': data['min_distance'],\n",
            "            'favorites': data['row'].get('favorites', 0),\n",
            "            'score': data['row'].get('score', 0),\n",
            "            'strategy': 'Multi-Query'\n",
            "        })\n",
            "        \n",
            "    df_res = pd.DataFrame(results)\n",
            "    \n",
            "    # Hybrid Scoring\n",
            "    import numpy as np\n",
            "    \n",
            "    # Normalize Frequency (Boost items that appear multiple times)\n",
            "    # We iterate N times, so max freq is N. \n",
            "    max_freq = df_res['frequency'].max()\n",
            "    df_res['freq_norm'] = df_res['frequency'] / max_freq\n",
            "    \n",
            "    # Normalize Similarity Sum (Average it? Or just use it as is?)\n",
            "    # Let's use sim_sum / frequency to get 'average similarity' of the triggers\n",
            "    df_res['avg_sim'] = df_res['sim_sum'] / df_res['frequency']\n",
            "    \n",
            "    # Popularity & Quality\n",
            "    df_res['fav_log'] = np.log1p(df_res['favorites'].fillna(0))\n",
            "    df_res['fav_norm'] = df_res['fav_log'] / (df_res['fav_log'].max() + 1e-6)\n",
            "    df_res['score_norm'] = df_res['score'].fillna(0) / 10.0\n",
            "    \n",
            "    # Final Score\n",
            "    # 40% Avg Similarity + 20% Frequency Bonus + 20% Popularity + 20% Quality\n",
            "    df_res['final_score'] = (\n",
            "        0.4 * df_res['avg_sim'] + \n",
            "        0.2 * df_res['freq_norm'] + \n",
            "        0.2 * df_res['fav_norm'] + \n",
            "        0.2 * df_res['score_norm']\n",
            "    )\n",
            "    \n",
            "    return df_res.sort_values('final_score', ascending=False).head(top_k)\n"
        ]

        # 3. Apply changes via iterating cells
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = cell.get("source", [])
                source_str = "".join(source)
                
                if "def recommend_centroid" in source_str:
                    cell["source"] = new_centroid_code
                    print("Updated recommend_centroid")
                    
                if "def recommend_multiquery" in source_str:
                    cell["source"] = new_multiquery_code
                    print("Updated recommend_multiquery")

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Notebook logic refined successfully.")

    except Exception as e:
        print(f"Error processing notebook: {e}")

if __name__ == "__main__":
    refine_notebook()
