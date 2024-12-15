import json
import pandas as pd
from sklearn.metrics import f1_score

# Define the file paths for the datasets
dataset_path_few_shot = r"eduqg_few_shot_bloom_cleaned.json"
dataset_path = r"eduqg_evaluation_bloom_cleaned.json"
blooms_taxonomy_path = r"Blooms_Taxonomy.csv"

# Bloom taxonomy categories
bloom_categories = ["Knowledge", "Comprehension", "Application", "Analysis"]

# Load Bloom's Taxonomy verbs
blooms_taxonomy_df = pd.read_csv(blooms_taxonomy_path)
blooms_taxonomy_verbs = blooms_taxonomy_df.to_dict(orient='list')

# Filter out nan values and convert all non-string values to strings
for key in blooms_taxonomy_verbs:
    blooms_taxonomy_verbs[key] = [str(verb) for verb in blooms_taxonomy_verbs[key] if pd.notna(verb)]

# Load the dataset
try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

# Function to classify a question based on the verbs
def classify_question(question, blooms_taxonomy_verbs):
    for category, verbs in blooms_taxonomy_verbs.items():
        for verb in verbs:
            if verb.lower() in question.lower():
                return category
    return "Unclassified"

# Classify the questions and calculate the F1 score
y_true = []
y_pred = []

for entry in dataset:
    for question_entry in entry["questions"]:
        question = question_entry["question"]["normal_format"]
        actual_bloom = question_entry["actual_bloom"]
        
        # Classify the question
        predicted_bloom = classify_question(question, blooms_taxonomy_verbs)
        
        if predicted_bloom != "Unclassified":
            y_true.append(bloom_categories.index(actual_bloom))
            y_pred.append(bloom_categories.index(predicted_bloom))

# Calculate the F1 score
if y_true and y_pred:
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use weighted average for imbalanced classes
    print(f"F1 Score: {f1}")
else:
    print("No valid predictions or ground truth values to calculate F1 score.")