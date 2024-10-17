import pandas as pd
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import Counter
import re

# Define input and output file paths
input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping Output data/Data_output 2.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM/Final Output2.xlsx"

# Initialize the Flair NER model
tagger = SequenceTagger.load("flair/ner-english-large")

# List of common location keywords for fallback pattern matching
common_location_keywords = ["USA", "China", "Japan", "India", "Russia", "Brazil", "Mexico", "UK", "France", "Germany",
                            "Australia", "Canada", "Italy", "Spain", "Iran", "Saudi Arabia", "South Korea", "Indonesia"]

# Dictionary to normalize location names (e.g., mapping US -> United States)
location_normalization_dict = {
    "us": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "k.k.": "united kingdom",
    "chinas": "china",
    # Add other common abbreviations and mappings here
}

def clean_content(content):
    """Remove unwanted phrases from the content and preprocess by removing special characters, converting to lowercase."""
    unwanted_phrases = [
        "View more news", "opens new tab", "Our Standards:", "The graph shows the current", "This article is more than",
        "Click here to view the list", "Gift 5 articles", "Subscribe", "Follow the topics",
        "Login", "Already a subscriber?", "Subscribe for all of The Times", "View Report", "Fetching latest articles",
        "Disclaimer", "While we try everything to ensure accuracy", "The designations employed and the presentation",
        "Copy link Copied Copy link Copied  to gift this article  to anyone you choose each month when you subscribe.",
        "(Reuters)"
    ]
    for phrase in unwanted_phrases:
        content = content.replace(phrase, "")
    content = re.sub(r'[^\w\s]', '', content)  # Remove special characters
    return content.lower().strip()


def normalize_location(location):
    """Normalize location names based on the predefined dictionary."""
    return location_normalization_dict.get(location, location)


def extract_locations_flair(text):
    """
    Use the Flair NER model to extract locations from text.
    """
    sentence = Sentence(text)
    tagger.predict(sentence)
    locations = [normalize_location(entity.text) for entity in sentence.get_spans('ner') if entity.tag == 'LOC']

    if locations:
        # Count the occurrences of each location
        location_counter = Counter(locations)
        # Return the most common location
        most_common_location = location_counter.most_common(1)[0][0]
        return most_common_location
    return None


def keyword_based_location_extraction(text):
    """
    A fallback method to extract locations based on common location keywords.
    """
    for keyword in common_location_keywords:
        if keyword.lower() in text.lower():
            return normalize_location(keyword)
    return None


def compare_and_finalize_location(title_location, content_location, keyword_location):
    """
    Compare locations from title, content, and keyword-based extraction, and return the most appropriate one.
    Priority: Content location > Title location > Keyword-based location.
    """
    if content_location:
        return content_location
    elif title_location:
        return title_location
    elif keyword_location:
        return keyword_location
    else:
        return 'Location not found'


def process_excel_file(input_file_path, output_file_path):
    """
    Read the input Excel file, extract locations from the title and content using both Flair NER and keyword-based methods,
    and write the results into a new Excel file.
    """
    # Read the input Excel file into a DataFrame
    df = pd.read_excel(input_file_path)

    # Initialize lists to store the final data
    titles, contents, finalized_locations = [], [], []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        title = row.get('Title', '')
        content = row.get('Content', '')

        # Clean the title and content
        cleaned_title = clean_content(str(title))
        cleaned_content = clean_content(str(content))

        # Extract locations using Flair NER from title and content
        location_from_title = extract_locations_flair(cleaned_title)
        location_from_content = extract_locations_flair(cleaned_content)

        # Extract locations using keyword-based method
        location_from_keywords = keyword_based_location_extraction(cleaned_content)

        # Compare and finalize the location
        finalized_location = compare_and_finalize_location(location_from_title, location_from_content,
                                                           location_from_keywords)

        # Append the results to the lists
        titles.append(title)
        contents.append(content)
        finalized_locations.append(finalized_location)

        print(f"Processed {index + 1}/{len(df)}: Final Location -> {finalized_location}")

    # Create a DataFrame with the final data
    result_df = pd.DataFrame({
        'Title': titles,
        'Content': contents,
        'Location': finalized_locations
    })

    # Write the DataFrame to an Excel file
    result_df.to_excel(output_file_path, index=False)
    print(f"Location extraction completed and saved to {output_file_path}")


# Call the function to process the Excel file
process_excel_file(input_file_path, output_file_path)