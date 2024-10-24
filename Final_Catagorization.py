from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import re
from sklearn.metrics import accuracy_score

# Input and output file paths
input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping data/Scrapping Output.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM data/LLM Output.xlsx"
true_category_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM Model/True Cat.xlsx"  # True category file

# Load the pre-trained FLAN-T5-large model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda" if torch.cuda.is_available() else "cpu")

# Predefined categories for classification with refined prompts
categories = ["Transportation", "Natural Disaster", "Geo-politics", "Trade", "Labor", "Others"]

# Content cleaning function
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
    content = re.sub(r'[^\w\s]', '', content)  # Remove special characters
    return content.lower().strip()

# Function to extract the most relevant location with a geo-political focus using FLAN-T5 (Generative AI)
def extract_relevant_location(content, max_length=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Refined prompt for location extraction, asking for the most relevant geopolitical location
    prompt = (
        f"Extract the most relevant location related to geo-political events from the following content. "
        f"If the content is geo-political, ensure the location includes country and state if applicable. "
        f"Return the country in standard form (e.g., United States instead of US): {content}"
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

    # Generate the location using the model
    outputs = model.generate(**inputs, max_new_tokens=30)  # Increased max_new_tokens to 30 for better context

    # Decode the generated location
    location = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return location if location else "Location not found"

# Function to generate category with more refined prompts for each category
def generate_category(content, max_length=512, num_samples=15, temperature=0.2):  # Reduced temperature to 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Refined prompt for category classification, focusing more on key signals for each category
    prompt = (
        f"Classify the following content into one of the categories: "
        f"Transport (content that specifically discusses the physical movement of goods or people via pipelines, shipping, air, rail, or road, or transport infrastructure), "
        f"Natural Disaster (content about natural events like earthquakes, floods, hurricanes, or wildfires, or the impact of such events on people or infrastructure), "
        f"Geo-politics (content focusing on international relations, military conflicts, defense policies, and diplomatic negotiations), "
        f"Trade (content involving business transactions, production, imports, exports, tariffs, trade agreements, stock market activity, corporate acquisitions, or economic sanctions), "
        f"Labour (content about worker-related issues, including strikes, labor unions, wage negotiations, layoffs, or changes in employment conditions), "
        f"Others (use this category for any content not related to the above categories).\n\n"
        f"Content: {content}\n"
        f"Respond with only the category name:"
    )

    # Tokenize the prompt and move inputs to the device (GPU/CPU)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

    # Generate multiple responses with temperature setting for more variety and confidence
    generated_responses = []
    for _ in range(num_samples):
        outputs = model.generate(**inputs, max_new_tokens=30, temperature=temperature, repetition_penalty=1.2)  # Increased max_new_tokens to 30
        result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generated_responses.append(result)

    # Post-process and find the most frequent category
    category_counts = {category: 0 for category in categories}
    for response in generated_responses:
        for category in categories:
            if category.lower() in response.lower():
                category_counts[category] += 1

    # Return the category with the highest count
    most_frequent_category = max(category_counts, key=category_counts.get)

    return most_frequent_category

# Function to process the Excel file using FLAN-T5 for both location extraction and content classification
def process_excel_file(input_file_path, output_file_path, true_category_file_path):
    df = pd.read_excel(input_file_path)
    true_category_df = pd.read_excel(true_category_file_path)  # Load true categories file

    titles, contents, finalized_locations, categories_list = [], [], [], []
    true_categories, predicted_categories = [], []  # For accuracy tracking

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        title = row.get('Title', '')
        content = row.get('Content', '')
        true_category = true_category_df.loc[index, 'Category']  # Get the true category from the true category file

        cleaned_title = clean_content(str(title))
        cleaned_content = clean_content(str(content))

        location_from_title = extract_relevant_location(cleaned_title)
        content_category = generate_category(cleaned_content)

        titles.append(title)
        contents.append(content)
        finalized_locations.append(location_from_title)
        categories_list.append(content_category)

        # For accuracy tracking
        true_categories.append(true_category)
        predicted_categories.append(content_category)

        print(f"Processed {index + 1}/{len(df)}: Location -> {location_from_title}, Category -> {content_category}")

    # Calculate and print accuracy for category classification
    category_accuracy = accuracy_score(true_categories, predicted_categories)
    print(f"Category classification accuracy: {category_accuracy * 100:.2f}%")

    # Create a DataFrame with the final data
    result_df = pd.DataFrame({
        'Title': titles,
        'Content': contents,
        'Location': finalized_locations,
        'Category': categories_list
    })

    # Write the DataFrame to an Excel file
    result_df.to_excel(output_file_path, index=False)
    print(f"Processing completed with Generative AI, saved to {output_file_path}")

# Run the processing function
process_excel_file(input_file_path, output_file_path, true_category_file_path)
