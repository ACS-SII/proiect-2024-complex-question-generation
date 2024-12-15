import json
import requests

# Define the endpoint and headers
url = "https://chat.readerbench.com/ollama/api/chat"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjM1MDcyODAwLTNhOGEtNDMwYy1hNjllLTRlZDU5NWYyZDdkOSJ9.YDwY5GK8eT_PNCsqb6ti7cjGwZEeoAkWG-tGBm5Tg-I",  # Replace with your actual API key
    "Content-Type": "application/json"
}

# Bloom taxonomy categories
bloom_categories = ["Knowledge", "Comprehension", "Application", "Analysis", "Evaluation", "Creation"]

# Function to classify a question using the model
def classify_question_with_model(question, context, choices):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that will help teachers to understand the complexity of the formulated questions and adapt the difficulty per student. Your task is to classify questions according to Bloom's Taxonomy. The categories are: Knowledge, Comprehension, Application, Analysis, Evaluation, Creation. Please provide the classification in the format: 'The Bloom's Taxonomy for this question is [Category]'. Think step-by-step before providing the final classification. This is a very important task, so take your time to think about it, please help me."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nChoices: {choices}"}
    ]

    data = {
        "model": "llama3.1:70b",
        "messages": messages
    }

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)

        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}: {response.text}")
        else:
            final_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode('utf-8'))
                        if "message" in json_line and "content" in json_line["message"]:
                            final_response += json_line["message"]["content"]
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse line as JSON: {line}")
            print("Final Response:", final_response.strip())

            if "The Bloom's Taxonomy for this question is" in final_response:
                classification = final_response.split("The Bloom's Taxonomy for this question is")[-1].strip()
                classification = classification.split('/')[0].strip()
                for category in bloom_categories:
                    if category.lower() in classification.lower():
                        return category
            else:
                for category in bloom_categories:
                    if category.lower() in final_response.lower():
                        return category

            print("Final Response:", final_response.strip())
            return "No Bloom category found"

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "Request failed"

# Example usage
if __name__ == "__main__":
    question = input("Enter the question: ")
    context = input("Enter the context: ")
    choices = input("Enter the choices (comma-separated): ").split(",")
    classification = classify_question_with_model(question, context, choices)
    print(f"The Bloom's Taxonomy for this question is: {classification}")