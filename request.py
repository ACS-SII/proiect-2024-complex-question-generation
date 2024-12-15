import json
import requests
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

# Define the file path for the dataset
dataset_path_few_shot = r"eduqg_few_shot_bloom_cleaned.json"
dataset_path = r"eduqg_evaluation_bloom_cleaned.json"
blooms_taxonomy_path = r"Blooms_Taxonomy.csv"
faulty_predictions_path = r"faulty_predictions.json"


# Define the endpoint and headers
url = "https://chat.readerbench.com/ollama/api/chat"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjM1MDcyODAwLTNhOGEtNDMwYy1hNjllLTRlZDU5NWYyZDdkOSJ9.YDwY5GK8eT_PNCsqb6ti7cjGwZEeoAkWG-tGBm5Tg-I",  # Replace with your actual API key
    "Content-Type": "application/json"
}

# Bloom taxonomy categories
bloom_categories = ["Knowledge", "Comprehension", "Application", "Analysis"]

# Load Bloom's Taxonomy verbs
blooms_taxonomy_df = pd.read_csv(blooms_taxonomy_path)
blooms_taxonomy_verbs = blooms_taxonomy_df.to_dict(orient='list')


# Load few-shot examples
few_shot_examples = []
try:
    with open(dataset_path_few_shot, 'r', encoding='utf-8') as f:
        few_shot_dataset = json.load(f)
            #few_shot_examples.append((question, bloom_type))
except Exception as e:
    print(f"Failed to load few-shot dataset: {e}")
    exit()

#Load the dataset and extract the first question
try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        #first_question = dataset[0]["questions"][0]["question"]["normal_format"] # Adjust the key if it's named differently
        #print("First Question:", first_question)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit()

# messages = [
#         {"role": "user", "content": "You are a helpful assistant. Your task is to classify questions according to Bloom's Taxonomy. The categories are: Knowledge , Comprehension, Application, Analysis. Please provide the classification in the format: 'The Bloom's Taxonomy for this question is [Category]'"},
#         # {"role": "user", "content": "Here are some examples of classifications: the context for the question is:  In the Hindu caste tradition , people were expected to work in the occupation of their caste and to enter into marriage according to their caste .   Accepting this social standing was considered a moral duty .  Cultural values reinforced the system . Caste systems promote beliefs in fate , destiny , and the will of a higher power , rather than promoting individual freedom as a value . A person who lived in a caste society was socialized to accept his or her social standing .  Caste systems are closed stratification systems in which people can do little or nothing to change their social standing .  A caste system is one in which people are born into their social standing and will remain in it their whole lives . People are assigned occupations regardless of their talents , interests , or potential . There are virtually no opportunities to improve one ' s social position . Sociologists distinguish between two types of systems of stratification .  Closed systems accommodate little change in social position .  They do not allow people to shift levels and do not permit social relations between levels . Open systems , which are based on achievement , allow movement and interaction between layers and classes . Different systems reflect , emphasize , and foster certain cultural values , and shape individual beliefs . Stratification systems include class systems and caste systems , as well as meritocracy, the question is: What factor makes caste systems closed?, the Bloom's Taxonomy for this question is Knowledge,\
#         #  the context for question is: In 1953 , Melvin Tumin countered the Davis-Moore thesis in \" Some Principles of Stratification : A Critical Analysis . \" Tumin questioned what determined a job ' s degree of importance . The Davis-Moore thesis does not explain , he argued , why a media personality with little education , skill , or talent becomes famous and rich on a reality show or a campaign trail .  The thesis also does not explain inequalities in the education system , or inequalities due to race or gender .   Tumin believed social stratification prevented qualified people from attempting to fill roles ( Tumin 1953 ) .   For example , an underprivileged youth has less chance of becoming a scientist , no matter how smart she is , because of the relative lack of opportunity available to her, the question is: Which of the following is correct about some qualified people and higher-level job positions?, the Bloom's Taxonomy for this question is Comprehension\
#         #  the context for the question is: Symbolic interactionism is a theory that uses everyday interactions of individuals to explain society as a whole .   Symbolic interactionism examines stratification from a micro-level perspective .   This analysis strives to explain how people ' s social standing affects their everyday interactions, the question is: Which statement represents stratification from the perspective of symbolic interactionism?, the Bloom's Taxonomy for this question is Application\
#         #  the context for the question is: Researchers also found that music can foster a sense of wholeness within a group . In fact , scientists who study the evolution of language have concluded that originally language ( an established component of group identity ) and music were one ( Darwin 1871 ) . Additionally , since music is largely nonverbal , the sounds of music can cross societal boundaries more easily than words . Music allows people to make connections where language might be a more difficult barricade .  As Fritz and his team found , music and the emotions it conveys can be cultural universals .  Music has the ability to evoke emotional responses .  In television shows , movies , even commercials , music elicits laughter , sadness , or fear .   Are these types of musical cues cultural universals ?, the question is: Most cultures have been found to identify laughter as a sign of humor, joy, or pleasure. Likewise, most cultures recognize music in some form. What are music and laughter are examples of?, the Bloom's Taxonomy for this question is Analysis"}
#     ]

# Filter out nan values and convert all non-string values to strings
for key in blooms_taxonomy_verbs:
    blooms_taxonomy_verbs[key] = [str(verb) for verb in blooms_taxonomy_verbs[key] if pd.notna(verb)]


messages = [
        {"role": "system", "content": "give me the bloom taxonomy level , respond like this the bloom taxonomy is"},

        #{"role": "system", "content": "You are a helpful assistant that will help teachers to understand the complexity of the formulated questions and adapt the difficulty per student. Your task is to classify questions according to Bloom's Taxonomy. The categories are: Knowledge, Comprehension, Application, Analysis, Evaluation, Creation. Please provide the classification in the format: 'The Bloom's Taxonomy for this question is [Category]'. Think step-by-step before providing the final classification. This is a very important task, so take your time to think about it, please help me."},
        # {"role": "system", "content": "Bloom's Taxonomy is a framework for categorizing educational goals. Here are more details about each category:"},
        # {"role": "system", "content": f"Knowledge: This category involves recalling facts, terms, basic concepts, and answers. Example verbs: {', '.join(blooms_taxonomy_verbs['Knowledge'])}."},
        # {"role": "system", "content": f"Comprehension: This category involves understanding the meaning of information, interpreting facts, comparing, contrasting, and explaining. Example verbs: {', '.join(blooms_taxonomy_verbs['Comprehension'])}."},
        # {"role": "system", "content": f"Application: This category involves using information in new situations, applying knowledge to solve problems, and using learned material in new and concrete situations. Example verbs: {', '.join(blooms_taxonomy_verbs['Application'])}."},
        # {"role": "system", "content": f"Analysis: This category involves breaking down information into parts, understanding its structure, and identifying motives or causes. It includes making inferences and finding evidence to support generalizations. Example verbs: {', '.join(blooms_taxonomy_verbs['Analysis'])}."},
        # {"role": "system", "content": f"Evaluation: This category involves making judgments based on criteria and standards. Example verbs: {', '.join(blooms_taxonomy_verbs['Evaluation'])}."},
        # {"role": "system", "content": f"Creation: This category involves putting elements together to form a coherent or functional whole. Example verbs: {', '.join(blooms_taxonomy_verbs['Creation'])}."}
    ]

for chapter in few_shot_dataset:
    for question_item in chapter.get('questions', []):
        question = question_item.get("question", {}).get("normal_format", "")
        actual_bloom = question_item.get("actual_bloom", {})
        context = question_item.get("hl_context_clean", "")
        choices = question_item.get("question", {}).get("question_choices", [])
        messages.append({"role": "user", "content": f"Context: {context}\nQuestion: {question}\nChoices: {choices}"})
    #messages.append({"role": "user", "content": "Question: {example_question}"})
    messages.append({"role": "assistant", "content": f"The Bloom's Taxonomy for this question is {actual_bloom}."})

# Define the request body using the first question
data = {
    #"model": "llama3.1:70b",
    "model": "llama3.2:3b",
    #"model": "qwen2.5:1.5b",
    "messages": messages
}
response = requests.post(url, headers=headers, json=data, stream=True)

# Function to classify a question and compare it with the actual Bloom 
# taxonomy
def get_bloom_classification(question, context, choices):

    messages = []
     # Include the context with the question
    #messages.append({"role": "user", "content": f"Context: {context}\nQuestion: {question}"})
    messages.append({"role": "user", "content": f"the context for question is: {context}\n, the question is: {question}\n, the choices are {choices}. Give me the bloom taxonomy level , respond like this the bloom taxonomy is "})



    # Define the request body using the first question
    data = {
        "model": "llama3.1:70b",
        #"model": "llama3.2:3b",
        #"model": "qwen2.5:1.5b",
        "messages": messages
    }

    # Initialize a variable to hold the concatenated response content
    final_response = ""

    # Make the POST request with streaming enabled
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)

        # Check if the response is successful
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}: {response.text}")
        else:
            # Stream and process each JSON object in the response
            for line in response.iter_lines():
                if line:  # Skip empty lines
                    try:
                        # Parse the JSON line
                        json_line = json.loads(line.decode('utf-8'))

                        # Append the "content" value if it exists
                        if "message" in json_line and "content" in json_line["message"]:
                            final_response += json_line["message"]["content"]
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse line as JSON: {line}")
            if "The Bloom's Taxonomy for this question is" in final_response:
                # Extract Bloom category from the response
                classification = final_response.split("The Bloom's Taxonomy for this question is")[-1].strip()
                classification = classification.split('/')[0].strip()
                #print(f"Question: {question}\nPredicted Bloom: {classification}")
                for category in bloom_categories:
                    if category.lower() in classification.lower():
                        return category
            elif "remembering" in final_response.lower():
                return "Knowledge"
                
            else:
                for category in bloom_categories:
                    if category.lower() in final_response.lower():
                        #print(f"Question: {question}\nFirst Bloom category found: {category}")
                        return category

            # Print the final response
            print("Final Response:", final_response.strip())

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# Evaluate F1 Score
y_true = []
y_pred = []
faulty_predictions = []

for chapter in dataset:
    for question_item in chapter.get('questions', []):
        question = question_item.get("question", {}).get("normal_format", "")
        actual_bloom = question_item.get("actual_bloom", {})
        context = question_item.get("hl_context_clean", "")
        choices = question_item.get("question", {}).get("question_choices", [])
        print(f"Question: {question}, context: {context}, choices: {choices}")
        print(f"Actual Bloom: {actual_bloom}")  

        # Skip if no question or actual bloom taxonomy
        if not question or not actual_bloom:
            continue
        
        # Get Bloom classification from the model
        predicted_bloom = get_bloom_classification(question, context, choices)
        #predicted_bloom = get_bloom_classification(question)

        print(f"Actual Bloom: {actual_bloom}, Predicted Bloom: {predicted_bloom}")
        
        if predicted_bloom:
            # Ensure classification is in the Bloom taxonomy categories
            if predicted_bloom in bloom_categories:
                y_true.append(bloom_categories.index(actual_bloom))
                y_pred.append(bloom_categories.index(predicted_bloom))
            
            # Save faulty predictions
            if predicted_bloom != actual_bloom:
                faulty_predictions.append({
                    "question": question,
                    "context": context,
                    "choices": choices,
                    "actual_bloom": actual_bloom,
                    "predicted_bloom": predicted_bloom
                })

# Save faulty predictions to a file
with open(faulty_predictions_path, 'w', encoding='utf-8') as f:
    json.dump(faulty_predictions, f, ensure_ascii=False, indent=4)


# Calculate the F1 score
if y_true and y_pred:
    f1 = f1_score(y_true, y_pred, average='weighted')  # Use weighted average for imbalanced classes
    print(f"F1 Score: {f1}")
else:
    print("No valid predictions or ground truth values to calculate F1 score.")