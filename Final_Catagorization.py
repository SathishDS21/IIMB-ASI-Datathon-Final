from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import re

input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping data/Scrapping Output.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM data/LLM Output.xlsx"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda" if torch.cuda.is_available() else "cpu")

categories = ["Transportation", "Natural Disaster", "Geo-politics", "Trade", "Labor", "Others"]

def clean_content(content):
    """Remove unwanted phrases from the content and preprocess by removing special characters, converting to lowercase."""
    unwanted_phrases = [
        "View more news", "opens new tab", "Our Standards:", "The graph shows the current", "This article is more than",
        "Click here to view the list", "Gift 5 articles", "Subscribe", "Follow the topics", "Login",
        "Already a subscriber?", "Subscribe for all of The Times", "View Report", "Fetching latest articles",
        "Disclaimer", "While we try everything to ensure accuracy", "Copy link Copied Copy link Copied"
    ]
    for phrase in unwanted_phrases:
        content = content.replace(phrase, "")
    content = re.sub(r'[^\w\s]', '', content) 
    return content.lower().strip()

def extract_relevant_location(content, max_length=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = (
        f"Extract the most relevant location related to geo-political events from the following content. "
        f"If the content is geo-political, ensure the location includes country and state if applicable. "
        f"Return the country in standard form (e.g., United States instead of US): {content}"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    location = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return location if location else "Location not found"

def generate_category(content, max_length=256, num_samples=10, temperature=0.4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = (
        f"Classify the following content into one of the categories: "
        f"Transport (includes pipelines, logistics, transport systems, vehicles, compressor stations, or activities like delivering or gathering gas), "
        f"Natural Disaster (includes floods, earthquakes, cyclones, hurricanes, or other natural events), "
        f"Geo-politics (international relations, government policies, conflicts), "
        f"Trade (activities like production, export, import, tariffs, business, index, commerce, trade deals, sanctions, selling, stocks, economic agreements, negotiations, funds, and economic exchanges), "
        f"Labor (pertaining to workers, strikes, labor unions, or employment-related matters), "
        f"Others (use this if none of the above categories apply).\n\n"
        f"Content: {content}\n"
        f"Respond with only the category name:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    generated_responses = []
    for _ in range(num_samples):
        outputs = model.generate(**inputs, max_new_tokens=5, temperature=temperature, repetition_penalty=1.2)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generated_responses.append(result)

    category_counts = {category: 0 for category in categories}
    for response in generated_responses:
        for category in categories:
            if category.lower() in response.lower():
                category_counts[category] += 1

    most_frequent_category = max(category_counts, key=category_counts.get)
    return most_frequent_category
    
def process_excel_file(input_file_path, output_file_path):
    df = pd.read_excel(input_file_path)
    titles, contents, finalized_locations, categories_list = [], [], [], []

    for index, row in df.iterrows():
        title = row.get('Title', '')
        content = row.get('Content', '')

        cleaned_title = clean_content(str(title))
        cleaned_content = clean_content(str(content))
        location_from_title = extract_relevant_location(cleaned_title)
        content_category = generate_category(cleaned_content)

        titles.append(title)
        contents.append(content)
        finalized_locations.append(location_from_title)
        categories_list.append(content_category)

        print(f"Processed {index + 1}/{len(df)}: Location -> {location_from_title}, Category -> {content_category}")

    result_df = pd.DataFrame({
        'Title': titles,
        'Content': contents,
        'Location': finalized_locations,
        'Category': categories_list
    })

    result_df.to_excel(output_file_path, index=False)
    print(f"Processing completed with Generative AI, saved to {output_file_path}")

process_excel_file(input_file_path, output_file_path)
