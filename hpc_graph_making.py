# disparity filter for extracting the multiscale backbone of complex weighted networks

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import polars as pl
import os
from nltk.tokenize import word_tokenize
from traceback import format_exception
import sys
from scipy.stats import percentileofscore
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx

project_path = os.getenv('BLACKHOLE') + "/SGAI-Final-Project/"

DEBUG = False

def disparity_integral (x, k):
    """
    calculate the definite integral for the PDF in the disparity filter
    """
    assert x != 1.0, "x == 1.0"
    assert k != 1.0, "k == 1.0"
    return ((1.0 - x)**k) / ((k - 1.0) * (x - 1.0))


def get_disparity_significance (norm_weight, degree):
    """
    calculate the significance (alpha) for the disparity filter
    """
    return 1.0 - ((degree - 1.0) * (disparity_integral(norm_weight, degree) - disparity_integral(0.0, degree)))

def report_error (cause_string, logger=None, fatal=False):
    """
    TODO: errors should go to logger, and not be fatal
    """
    etype, value, tb = sys.exc_info()
    error_str = "{} {}".format(cause_string, str(format_exception(etype, value, tb, 3)))

    if logger:
        logger.info(error_str)
    else:
        print(error_str)

    if fatal:
        sys.exit(-1)

def disparity_filter (graph):
    """
    implements a disparity filter, based on multiscale backbone networks
    https://arxiv.org/pdf/0904.2389.pdf
    """
    alpha_measures = []
    
    for node_id in graph.nodes():
        node = graph.nodes[node_id]
        degree = graph.degree(node_id)
        strength = 0.0

        # Use graph.edges to handle undirected edges
        for id0, id1 in graph.edges(node_id):
            edge = graph[id0][id1]
            strength += edge["weight"]

        node["strength"] = strength

        for id0, id1 in graph.edges(node_id):
            edge = graph[id0][id1]

            norm_weight = edge["weight"] / strength
            edge["norm_weight"] = norm_weight

            if degree > 1:
                try:
                    if norm_weight == 1.0:
                        norm_weight -= 0.0001

                    alpha = get_disparity_significance(norm_weight, degree)
                except AssertionError:
                    report_error("disparity {}".format(repr(node)), fatal=True)

                edge["alpha"] = alpha
                alpha_measures.append(alpha)
            else:
                edge["alpha"] = 0.0

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]
        edge["alpha_ptile"] = percentileofscore(alpha_measures, edge["alpha"]) / 100.0

    return alpha_measures

# related metrics

def calc_quantiles (metrics, num):
    """
    calculate `num` quantiles for the given list
    """
    global DEBUG

    bins = np.linspace(0, 1, num=num, endpoint=True)
    s = pd.Series(metrics)
    q = s.quantile(bins, interpolation="nearest")

    try:
        dig = np.digitize(metrics, q) - 1
    except ValueError as e:
        print("ValueError:", str(e), metrics, s, q, bins)
        sys.exit(-1)

    quantiles = []

    for idx, q_hi in q.iteritems():
        quantiles.append(q_hi)

        if DEBUG:
            print(idx, q_hi)

    return quantiles

def cut_graph (graph, min_alpha_ptile=0.5, min_degree=2):
    """
    apply the disparity filter to cut the given graph
    """
    filtered_set = set([])

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]

        if edge["alpha_ptile"] < min_alpha_ptile:
            filtered_set.add((id0, id1))

    for id0, id1 in filtered_set:
        graph.remove_edge(id0, id1)

    filtered_set = set([])

    for node_id in graph.nodes():
        node = graph.nodes[node_id]

        if graph.degree(node_id) < min_degree:
            filtered_set.add(node_id)

    for node_id in filtered_set:
        graph.remove_node(node_id)


with open(project_path + 'flight_network_graph.pickle', 'rb') as f:
   G = pickle.load(f)



# Initialize an empty DataFrame with the required columns and explicit types
text_df = pl.DataFrame({"node": pl.Series([], dtype=pl.Utf8),
                        "history": pl.Series([], dtype=pl.Utf8),
                        "demography": pl.Series([], dtype=pl.Utf8)})

# Process files in the `history` folder
for filename in os.listdir(project_path + 'history'):
    filename_wo_txt = os.path.splitext(filename)[0]
    if filename_wo_txt in G.nodes():
        with open(os.path.join(project_path + 'history', filename), 'r', encoding='utf-8') as file:
            content = file.read()
        new_row = pl.DataFrame({"node": [filename_wo_txt], "history": [content], "demography": [None]})
        text_df = pl.concat([text_df, new_row], how="vertical", rechunk=True)

# Process files in the `demography` folder
for filename in os.listdir(project_path + 'demography'):
    filename_wo_txt = os.path.splitext(filename)[0]
    if filename_wo_txt in G.nodes():
        with open(os.path.join(project_path + 'demography', filename), 'r', encoding='utf-8') as file:
            content = file.read()
        new_row = pl.DataFrame({"node": [filename_wo_txt], "history": [None], "demography": [content]})
        text_df = pl.concat([text_df, new_row], how="vertical", rechunk=True)

# Combine rows for the same `node` manually
nodes = text_df["node"].unique()
aggregated_data = []

for node in nodes:
    node_data = text_df.filter(text_df["node"] == node)
    history = node_data["history"].drop_nulls().to_list()[0] if not node_data["history"].drop_nulls().is_empty() else None
    demography = node_data["demography"].drop_nulls().to_list()[0] if not node_data["demography"].drop_nulls().is_empty() else None
    aggregated_data.append({"node": node, "history": history, "demography": demography})

text_df = pl.DataFrame(aggregated_data)


history_df = text_df.select(["node", "history"])

# Ensure the "history" column is of type string (Utf8) for further processing
history_df = history_df.with_columns(pl.col("history").cast(pl.Utf8))

# Filter out rows where the "history" column is null to only have valid content
history_df = history_df.filter(pl.col("history").is_not_null())


# get all the tokens that refer to the same genre
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
#stop_words.update(['http','title', 'short', 'description', ' ', 'wa'])

city_history_dict = {}
for row in history_df.iter_rows(named=True):
    city = row["node"]
    city_text = row["history"]

    # Tokenize text
    tokens = word_tokenize(city_text)
    # Convert to lowercase and remove punctuation
    tokens = [word.lower() for word in tokens if word.isalnum()]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # remove single characters
    tokens = [word for word in tokens if len(word) > 1]

    content = " ".join(tokens)
    city_history_dict[city] = content


#find number of subsections in text file
def find_all_sections(wiki_content):
    # Match any section with = signs (e.g., ===Section===) and its content
    pattern = r'(={2,}\s*[^=\n]+\s*={2,})([\s\S]*?)(?=\n={2,}[^=]+=*|\Z)'
    matches = re.findall(pattern, wiki_content)

    # Extract section headers and their content into a dictionary
    sections = {}
    for header, content in matches:
        # Clean the header of excess = signs and whitespace
        clean_header = re.sub(r'^={2,}\s*|\s*={2,}$', '', header).strip()
        sections[clean_header] = content.strip()

    return sections


cities = history_df['node'].to_list()

sub_sections = {}
# Get subsection count of each city
for city in cities:
    path = project_path + f'history/{city}.txt'
    with open(path, 'r', encoding='utf-8') as file:
        wiki_content = file.read()
    sections = find_all_sections(wiki_content)
    sub_sections[city] = len(sections)

# Get mean
mean_sub_sections = round(sum(sub_sections.values()) / len(sub_sections))
print(f"Mean number of subsections per city: {mean_sub_sections}")

# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Apply the vectorizer to the city history text data to create a term frequency matrix
tf = vectorizer.fit_transform(city_history_dict.values())

# Initialize the Latent Dirichlet Allocation (LDA) model to extract 10 topics
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(tf)

topics = lda.components_  # Word distributions per topic

# Transform the term frequency matrix to get the topic distribution for each city
city_topics = lda.transform(tf)


cosine_similarities_lda = cosine_similarity(city_topics)
cosine_similarities_lda = pd.DataFrame(cosine_similarities_lda, index=cities, columns=cities)

print("Mean: ", cosine_similarities_lda.mean().mean())
print(f'{cosine_similarities_lda.mean().idxmax()} is the city with the most similarities with an average of {cosine_similarities_lda.mean().max()}')
print(f'{cosine_similarities_lda.mean().idxmin()} is the city with the least similarities with an average of {cosine_similarities_lda.mean().min()}')
# Get the two cities with the highest similarities besides their own

print("Max: ", cosine_similarities_lda.mean().max())
print("Min: ", cosine_similarities_lda.mean().min())
print("Median: ", cosine_similarities_lda.mean().median())
print("Std: ", cosine_similarities_lda.mean().std())


print("Starting to apply disperity filter")
G_hist = nx.Graph()

for city1 in cosine_similarities_lda.index:
    for city2 in cosine_similarities_lda.columns:
        if city1 != city2:  # Exclude self-loops
            weight = cosine_similarities_lda.loc[city1, city2]
            if weight > 0:  # Optional: Add a threshold to include only significant weights
                G_hist.add_edge(city1, city2, weight=weight)


G_disparity = G_hist.copy()

alpha_measures = disparity_filter(G_disparity)
print("Alpha measures calculated:", alpha_measures)

quantiles = calc_quantiles(alpha_measures, num = 10)

# Determine alpha cutoff (median, 30th percentile)
min_alpha_ptile = 0.95
alpha_cutoff = quantiles[int(min_alpha_ptile * (len(quantiles) - 1))]
print(f"Using alpha cutoff at {min_alpha_ptile * 100}% percentile: {alpha_cutoff}")

# Filter the graph based on the cutoff
cut_graph(G_disparity, min_alpha_ptile=min_alpha_ptile)

print(f"There were {len(G_hist.nodes())} nodes and {len(G_hist.edges())} edges in un-filtered graph")
print(f"There are currently {len(G_disparity.nodes())} nodes and {len(G_disparity.edges())} edges in filtered graph")

# Save the graph to a file using pickle
with open(project_path + 'graph_hpc.pickle', 'wb') as f:
   pickle.dump(G, f)
   
print("Done!")
