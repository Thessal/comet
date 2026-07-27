#!../.venv/bin/python 
import pandas as pd 
import re
import json
from collections import defaultdict
import argparse  # TODO

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze agent convergence and divergence from log files.")
    parser.add_argument("filename", nargs="?", default="output_4_4", help="Filename or prefix to analyze (e.g., output_4_4)")
    return parser.parse_args()

# filename = "output_4_4"
def read(filename):
    # cat output_4_3.txt | grep -oE "(pool|Iteration)".* > output_4_3.tmp
    with open(f"../{filename}.tmp", "rt") as f: 
    # with open("../output_4_3.tmp", "rt") as f: 
        lines = f.readlines()
    iteration = 0 
    agent = -1
    equations = []
    for line in lines:
        if line.startswith("Iteration"):
            _, iteration, _, agent, _ = line.split()
            iteration = int(iteration)
            agent = int(agent.replace(")", ""))
        elif line.startswith("pool"):
            eqn = line.replace("pool : ", "").strip()
            equations.append({"iteration": iteration, "agent":agent, "equation":eqn})

    return pd.DataFrame(equations)

def build_data_category_map():
    with open("../wrds/category.json", "rt") as f :
        field_map = json.load(f)
    # field_map["price"] = ["price", "high", 'low', 'volume', 'close', 'returns']
    for f in ["price", "high", "low", "volume", "close", "returns"]:
        field_map[f] = [f]
    field_map.pop("_note", None)
    inverse_field_map = {field: k for k,v in field_map.items() for field in v}
    return defaultdict(lambda : "unknown", inverse_field_map)

# def build_data_category_map_():
#     field_map = {f: [f] for f in ["price", "high", "low", "volume", "close", "returns"]}
#     inverse_field_map = {field: k for k, v in field_map.items() for field in v}
#     return defaultdict(lambda: "unknown", inverse_field_map)

def analyze(filename):
    df = read(filename)
    field_map = build_data_category_map()
    
    # Complexity: roughly the number of open parenthesis
    df['complexity'] = df['equation'].apply(lambda x: x.count('('))
    
    # Extract data fields
    def get_categories(eqn):
        fields = re.findall(r'data\("([^"]+)"\)', eqn)
        return [field_map[f.split("_")[-1]] for f in fields]
    
    df['categories'] = df['equation'].apply(get_categories)
    
    # We bin iterations by 100 to get a clearer picture over time
    bin_size = 100
    df['iteration_bin'] = (df['iteration'] // bin_size) * bin_size
    
    # --- 1. Complexity Convergence ---
    # Calculate average complexity per agent in the bin
    agent_complexity = df.groupby(['iteration_bin', 'agent'])['complexity'].mean().reset_index()
    # Look across all agents: mean complexity, and standard deviation (divergence)
    complexity_trend = agent_complexity.groupby('iteration_bin')['complexity'].agg(['mean', 'std']).reset_index()
    complexity_trend.rename(columns={'mean': 'avg_complexity', 'std': 'std_complexity_across_agents'}, inplace=True)
    complexity_trend['std_complexity_across_agents'] = complexity_trend['std_complexity_across_agents'].fillna(0).round(2)
    complexity_trend['avg_complexity'] = complexity_trend['avg_complexity'].round(2)
    
    # --- 2. Category Convergence ---
    df_exploded = df.explode('categories').dropna(subset=['categories'])
    if not df_exploded.empty:
        # Count occurrences of categories per bin
        cat_counts = df_exploded.groupby(['iteration_bin', 'categories']).size().reset_index(name='count')
        
        # Get top 2 categories for context
        cat_counts = cat_counts.sort_values(by=['iteration_bin', 'count'], ascending=[True, False])
        top_cats = cat_counts.groupby('iteration_bin').head(2)
        top_cats_str = top_cats.groupby('iteration_bin').apply(
            lambda x: ', '.join([f"{row['categories']}({row['count']})" for _, row in x.iterrows()])
        ).reset_index(name='top_categories')
        
        # Calculate divergence in category space:
        agent_cat_counts = df_exploded.groupby(['iteration_bin', 'agent', 'categories']).size().reset_index(name='count')
        agent_totals = agent_cat_counts.groupby(['iteration_bin', 'agent'])['count'].sum().reset_index(name='total')
        agent_cat_counts = agent_cat_counts.merge(agent_totals, on=['iteration_bin', 'agent'])
        agent_cat_counts['ratio'] = agent_cat_counts['count'] / agent_cat_counts['total']
        
        # Standard deviation of ratios across agents for each category
        # First we need to make sure 0 counts exist if an agent doesn't use a category used by others
        # simpler: just std on what we have, fillna(0) for missing?
        # Actually to be completely rigorous we should pivot, but let's just group by categories and take std over available agents
        # (Assuming agents missing means ratio is 0, doing simple std without filling 0s isn't perfectly accurate, but good enough for trending)
        cat_std_across_agents = agent_cat_counts.groupby(['iteration_bin', 'categories'])['ratio'].std().fillna(0).reset_index(name='std')
        
        # Sum of stds -> roughly how divergent agent preferences are
        divergence_score = cat_std_across_agents.groupby('iteration_bin')['std'].sum().reset_index(name='category_divergence_score')
        divergence_score['category_divergence_score'] = divergence_score['category_divergence_score'].round(2)
        
        # Calculate average agent correlation/distance metrics
        import numpy as np
        from scipy.spatial.distance import pdist
        from scipy.special import rel_entr, kl_div
        
        def calc_agent_metrics(group):
            # Pivot to get category counts for each agent
            pivot = group.pivot(index='agent', columns='categories', values='count').fillna(0)
            if len(pivot) < 2 or pivot.shape[1] == 0:
                return pd.Series({
                    'avg_correlation': np.nan,
                    'avg_cosine_sim': np.nan,
                    'avg_euclidean_dist': np.nan,
                    'avg_kl_divergence': np.nan
                })
            
            # Normalize to proportions
            proportions = pivot.div(pivot.sum(axis=1), axis=0)
            
            # Laplace / additive smoothing for KL & JS divergence calculation to avoid 0-count extreme artifacts
            alpha = 1.0  # Add-1 smoothing
            vocab_size = pivot.shape[1]
            smoothed_counts = pivot + alpha
            smoothed_proportions = smoothed_counts.div(smoothed_counts.sum(axis=1), axis=0)
            
            n = len(pivot)
            
            # Pearson correlation
            corr_matrix = proportions.T.corr(method='pearson')
            if n > 1 and not corr_matrix.empty:
                upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
                if len(upper_tri) > 0 and not np.isnan(upper_tri).all():
                    avg_corr = np.nanmean(upper_tri)
                else:
                    avg_corr = np.nan
            else:
                avg_corr = np.nan
                
            # Cosine similarity (1 - cosine distance)
            try:
                cosine_dist = pdist(proportions.values, metric='cosine')
                avg_cos = 1.0 - np.nanmean(cosine_dist)
            except Exception:
                avg_cos = np.nan
                
            # Euclidean distance
            try:
                euclid_dist = pdist(proportions.values, metric='euclidean')
                avg_euclid = np.nanmean(euclid_dist)
            except Exception:
                avg_euclid = np.nan
                
            # KL divergence and Jensen-Shannon Divergence (JSD) for all possible agent pairs
            kl_vals = []
            jsd_vals = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        p = smoothed_proportions.iloc[i].values
                        q = smoothed_proportions.iloc[j].values
                        # Pairwise KL divergence
                        kl_vals.append(np.sum(rel_entr(p, q)))
                        # kl_vals.append(np.sum(kl_div(p, q)))
                        # print( pdist(np.array([p,q]), metric='cosine') )
                        # kl_vals.append( pdist(np.array([p,q]), metric='cosine')[0] )
                        # Pairwise Jensen-Shannon Divergence (JSD)
                        # print(np.round(p,2), np.round(q,2))
                        m = 0.5 * (p + q)
                        jsd = 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
                        jsd_vals.append(jsd)

            # avg_kl = np.nanmean(kl_vals) if kl_vals else np.nan
            avg_kl = np.nanmax(kl_vals) if kl_vals else np.nan
            avg_jsd = np.nanmean(jsd_vals) if jsd_vals else np.nan
                
            return pd.Series({
                'avg_correlation': round(avg_corr, 2),
                'avg_cosine_sim': round(avg_cos, 2),
                'avg_euclidean_dist': round(avg_euclid, 2),
                'max_kl_divergence': round(avg_kl, 2),
                'avg_js_divergence': round(avg_jsd, 4)
            })
            
        # print(agent_cat_counts.iloc[:10])
        agent_metrics = agent_cat_counts.groupby('iteration_bin').apply(calc_agent_metrics).reset_index()
        
        # Calculate sampled KL divergence from two mutually exclusive random sampled equation sets
        def calc_sampled_kl(group_df):
            n_eqs = len(group_df)
            if n_eqs < 2:
                return np.nan
            
            # Extract categories and their overall set in this bin
            all_cats_in_bin = list(set([c for cats in group_df['categories'] for c in cats]))
            if not all_cats_in_bin:
                return np.nan
                
            shuffled = group_df.sample(frac=1.0, random_state=42)
            split_point = n_eqs // 2
            set_a = shuffled.iloc[:split_point]
            set_b = shuffled.iloc[split_point:]
            
            cats_a = [c for cats in set_a['categories'] for c in cats]
            cats_b = [c for cats in set_b['categories'] for c in cats]
            
            from collections import Counter
            counts_a = Counter(cats_a)
            counts_b = Counter(cats_b)
            
            epsilon = 1e-6
            total_a = len(cats_a) + epsilon * len(all_cats_in_bin)
            total_b = len(cats_b) + epsilon * len(all_cats_in_bin)
            
            kl = 0.0
            for cat in all_cats_in_bin:
                p = (counts_a[cat] + epsilon) / total_a
                q = (counts_b[cat] + epsilon) / total_b
                kl += p * np.log(p / q)
            return round(kl, 2)
            
        sampled_kls = df.groupby('iteration_bin').apply(calc_sampled_kl).reset_index(name='sampled_kl_divergence')

        # --- 3. Silhouette Score per Iteration Bin ---
        from scipy.spatial.distance import cdist

        def calc_silhouette(bin_group):
            # Category count vector per equation
            all_categories = sorted(list(set([c for cats in bin_group['categories'] for c in cats])))
            if len(all_categories) == 0:
                return np.nan
            
            cat_to_idx = {c: i for i, c in enumerate(all_categories)}
            
            eq_vectors = []
            labels = []
            for _, row in bin_group.iterrows():
                if not row['categories']:
                    continue
                vec = np.zeros(len(all_categories))
                from collections import Counter
                counts = Counter(row['categories'])
                for c, cnt in counts.items():
                    vec[cat_to_idx[c]] = cnt
                
                total_cnt = vec.sum()
                if total_cnt > 0:
                    vec = vec / total_cnt
                eq_vectors.append(vec)
                labels.append(row['agent'])

            if len(eq_vectors) < 2:
                return np.nan
            
            X = np.array(eq_vectors)
            labels = np.array(labels)

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_samples = len(labels)

            if n_clusters < 2 or n_clusters >= n_samples:
                return np.nan

            try:
                dist_matrix = cdist(X, X, metric='euclidean')
                # Replace any potential nan/inf from zero vectors
                dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)

                sil_scores = []
                for i in range(n_samples):
                    own_cluster = labels[i]
                    same_cluster_mask = (labels == own_cluster)
                    same_cluster_mask[i] = False
                    
                    if not np.any(same_cluster_mask):
                        # Single-element cluster
                        a_i = 0.0
                    else:
                        a_i = dist_matrix[i, same_cluster_mask].mean()

                    b_i = np.inf
                    for other_cluster in unique_labels:
                        if other_cluster == own_cluster:
                            continue
                        other_mask = (labels == other_cluster)
                        mean_dist = dist_matrix[i, other_mask].mean()
                        if mean_dist < b_i:
                            b_i = mean_dist

                    if np.isinf(b_i) or max(a_i, b_i) == 0:
                        s_i = 0.0
                    else:
                        s_i = (b_i - a_i) / max(a_i, b_i)
                    sil_scores.append(s_i)

                return round(np.mean(sil_scores), 4)
            except Exception:
                return np.nan

        silhouette_trends = df.groupby('iteration_bin').apply(calc_silhouette).reset_index(name='silhouette_score')
        
        final_trend = complexity_trend.merge(divergence_score, on='iteration_bin', how='left')
        final_trend = final_trend.merge(agent_metrics, on='iteration_bin', how='left')
        final_trend = final_trend.merge(sampled_kls, on='iteration_bin', how='left')
        final_trend = final_trend.merge(silhouette_trends, on='iteration_bin', how='left')
        final_trend = final_trend.merge(top_cats_str, on='iteration_bin', how='left')
    else:
        final_trend = complexity_trend
        
    return final_trend, df

if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    filename = parse_args().__dict__["filename"]
    
    trend, raw_df = analyze(filename)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("=== Agent Convergence/Divergence Over Time (Binned by 100 Iterations) ===")
    print("Notes:")
    print(" - avg_complexity: Mean length/complexity of equations in this bin.")
    print(" - std_complexity_across_agents: Higher value = agents diverge in complexity. 0 = total convergence.")
    print(" - category_divergence_score: Higher value = agents prefer completely divergent fields.")
    print(" - avg_correlation: Average pairwise Pearson correlation of category usage between agents (higher = more correlated).")
    print(" - avg_cosine_sim: Average pairwise Cosine similarity of category usage between agents (higher = more similar).")
    print(" - avg_euclidean_dist: Average pairwise Euclidean distance of category usage vectors between agents (lower = closer).")
    print(" - avg_kl_divergence: (a) Average KL divergence of category distributions for all ordered agent pairs (divergence across agents, Laplace smoothed).")
    print(" - avg_js_divergence: Average Jensen-Shannon Divergence (JSD) across agent pairs (bounded, symmetric).")
    print(" - sampled_kl_divergence: (b) KL divergence of category distributions between two mutually exclusive random sampled sets of equations from the bin (baseline divergence).")
    print(" - silhouette_score: Silhouette score of equations across agents using cosine distance (higher = agents form distinct clusters).")
    print(" - top_categories: Data fields favored by the pool in this iteration block.")
    print("-" * 210)
    print(trend.to_string(index=False))
    print(trend.iloc[:, :-1].to_string())

    # --- Generate Silhouette Plots for First and Last Batches ---
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    from collections import Counter
    import numpy as np

    bins = sorted(raw_df['iteration_bin'].unique())
    first_bin = bins[0]
    # last_bin = bins[-1]
    last_bin = bins[-2] # last bin may be not complete


    def compute_sample_silhouettes(bin_df):
        all_categories = sorted(list(set([c for cats in bin_df['categories'] for c in cats])))
        if len(all_categories) == 0:
            return np.array([]), np.array([]), 0.0

        cat_to_idx = {c: i for i, c in enumerate(all_categories)}
        eq_vectors, labels = [], []
        for _, row in bin_df.iterrows():
            if not row['categories']:
                continue
            vec = np.zeros(len(all_categories))
            counts = Counter(row['categories'])
            for c, cnt in counts.items():
                vec[cat_to_idx[c]] = cnt
            total_cnt = vec.sum()
            if total_cnt > 0:
                vec = vec / total_cnt
            eq_vectors.append(vec)
            labels.append(row['agent'])

        if len(eq_vectors) < 2:
            return np.array([]), np.array([]), 0.0

        X = np.array(eq_vectors)
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        n_samples = len(labels)

        dist_matrix = cdist(X, X, metric='euclidean')
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)

        sil_scores = []
        for i in range(n_samples):
            own_cluster = labels[i]
            same_cluster_mask = (labels == own_cluster)
            same_cluster_mask[i] = False

            if not np.any(same_cluster_mask):
                a_i = 0.0
            else:
                a_i = dist_matrix[i, same_cluster_mask].mean()

            b_i = np.inf
            for other_cluster in unique_labels:
                if other_cluster == own_cluster:
                    continue
                other_mask = (labels == other_cluster)
                mean_dist = dist_matrix[i, other_mask].mean()
                if mean_dist < b_i:
                    b_i = mean_dist

            if np.isinf(b_i) or max(a_i, b_i) == 0:
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)
            sil_scores.append(s_i)

        return np.array(sil_scores), labels, np.mean(sil_scores)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

    for ax, bin_val, title in zip(axes, [first_bin, last_bin], [f"First Batch (Iteration Bin {first_bin})", f"Last Batch (Iteration Bin {last_bin})"]):
        bin_df = raw_df[raw_df['iteration_bin'] == bin_val]
        sample_sil_values, labels, avg_score = compute_sample_silhouettes(bin_df)

        if len(sample_sil_values) == 0:
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
            continue

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(sample_sil_values) + (n_clusters + 1) * 10])

        y_lower = 10
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

        for i, cluster_id in enumerate(unique_labels):
            ith_cluster_sil_values = sample_sil_values[labels == cluster_id]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = colors[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_sil_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
                label=f"Agent {cluster_id}"
            )
            ax.text(-0.95, y_lower + 0.5 * size_cluster_i, f"Ag {cluster_id}", fontsize=8, va='center')
            y_lower = y_upper + 10

        ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Avg Score: {avg_score:.3f}")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Silhouette Coefficient Values (Euclidean Distance)")

    axes[0].set_ylabel("Equations (Grouped by Agent)")
    plt.tight_layout()
    plot_path = f"silhouette_comparison_{filename}.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved silhouette comparison plot to {plot_path}")