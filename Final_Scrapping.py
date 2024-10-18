import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import re
import time
import random

# Define the input and output file paths
input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping data/Scrapping Input.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping data/Scrapping Output.xlsx"

# List of unwanted words/phrases to remove from the scraped content (duplicates removed)
unwanted_phrases = list(set([
    "View more news", "opens new tab", "Our Standards:", "The graph shows the current", "This article is more than",
    "Click here to view the list", "Gift 5 articles", "Subscribe", "Follow the topics",
    "Login", "Already a subscriber?", "Subscribe for all of The Times", "View Report", "Fetching latest articles",
    "Disclaimer", "While we try everything to ensure accuracy",
    "The designations employed and the presentation of material on the map",
    "more taiwan news  2024 all rights reserved  ",
    "Copy link Copied Copy link Copied  to gift this article  to anyone you choose each month when you subscribe.",
    "(Reuters)",
    # Add more unwanted phrases here as needed
]))

def clean_content(content):
    """Remove unwanted phrases from the content and preprocess by removing special characters, converting to lowercase."""
    for phrase in unwanted_phrases:
        content = content.replace(phrase, "")

    # Remove special characters and convert to lowercase
    content = re.sub(r'[^\w\s]', '', content)  # Remove special characters
    content = content.lower()  # Convert to lowercase
    return content.strip()

def scrape_with_cloudscraper(url):
    """Scrape using CloudScraper and target the main article section, checking multiple patterns."""
    attempts = 3  # Number of attempts for retry mechanism
    for attempt in range(attempts):
        try:
            print(f"Attempting with CloudScraper for URL: {url}")

            # Define the headers to simulate a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            # Create a CloudScraper instance
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, headers=headers)

            # Check if the response was successful (status code 200)
            if response.status_code == 200:
                # Parse the HTML response with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract the title
                title = scrape_title(soup)
                if not title:
                    print("Title not found!")
                    return None

                # Check if the title contains "Event summary" and skip if it does
                if "event summary" in title.lower():  # Convert to lowercase for case-insensitive matching
                    print(f"Skipping URL: {url} because it contains 'Event summary'")
                    return None

                # Extract the content
                content = scrape_content(soup)
                if not content:
                    print("Content not found!")
                    return None

                # Clean the content by removing unwanted phrases, special characters, and converting to lowercase
                cleaned_content = clean_content(content)

                # Clean the title in a similar way (remove special characters and convert to lowercase)
                cleaned_title = clean_content(title)



                # Return the scraped data
                return {"title": cleaned_title, "content": cleaned_content}
            else:
                print(f"CloudScraper failed with status code: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error with CloudScraper: {e}")
            if attempt < attempts - 1:
                wait_time = random.uniform(1, 3)  # Random wait time between 1 to 3 seconds
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    return None  # Return None if all attempts fail

# The rest of your functions remain the same

def scrape_title(soup):
    """Try multiple patterns to extract the title."""
    title_patterns = [
        {'tag': 'h1', 'attr': {'class': '_804759db85ec8cc17e4f', 'data-testid': 'ArticleHeader-headline'}},
        {'tag': 'div', 'attr': {'id': 'alert_TC_Green_title_big'}},  # Supply risk title
        {'tag': 'span', 'attr': {'id': 'ctl00_masterReportTitle'}},  # Another supply risk pattern
        {'tag': 'h1'},  # Generic <h1> tag
        {'tag': 'h2'},  # Generic <h2> tag for some summaries
        {'tag': 'div', 'attr': {'class': 'view_headline LoraMedium'}},  # Korean Times specific
        {'tag': 'div', 'attr': {'data-qa': 'headline'}},  # Other website-specific titles
        {'tag': 'h1', 'attr': {'class': 'css-xyz'}},  # Other sites
        {'tag': 'div', 'attr': {'class': 'alert_title'}},  # Corrected pattern for supply risk
    ]

    for pattern in title_patterns:
        tag = pattern.get('tag')
        attr = pattern.get('attr', {})
        title_element = soup.find(tag, attr)
        if title_element:
            return title_element.text.strip()

    return None  # Return None if no title is found

def scrape_content(soup):
    """Try multiple patterns to extract the content."""
    content_patterns = [
        {'tag': 'div', 'attr': {'data-testid': lambda x: x and x.startswith('paragraph-')}},
        {'tag': 'p'},  # Generic <p> tag for AFR and other websites
        {'tag': 'td', 'attr': {'class': 'cell_value_summary'}},
        {'tag': 'span', 'attr': {'class': 'read'}},  # Korean Times content
        {'tag': 'p', 'attr': {'class': 'p_summary'}},  # <p class="p_summary"> pattern
        {'tag': 'p', 'attr': {'class': 'css-at9mc1 evys1bk0'}},  # NYTimes style
    ]

    for pattern in content_patterns:
        tag = pattern.get('tag')
        attr = pattern.get('attr', {})
        elements = soup.find_all(tag, attr)
        if elements:
            content = " ".join([el.text.strip() for el in elements])
            return content

    return None  # Return None if no content is found

def scrape_from_excel(input_file_path, output_file_path):
    """Read URLs from input Excel file, scrape data, and write results to an output Excel file."""
    df = pd.read_excel(input_file_path)

    titles, contents, locations = [], [], []

    for index, row in df.iterrows():
        url = row['Links']  # Assuming the column containing URLs is named 'Links'
        print(f"Scraping URL {index + 1}: {url}")

        result = scrape_with_cloudscraper(url)

        if result:
            titles.append(result['title'])
            contents.append(result['content'])

        else:
            titles.append("")
            contents.append("")


        wait_time = random.uniform(1, 5)  # Random wait time between 1 to 5 seconds
        print(f"Waiting for {wait_time:.2f} seconds before the next request...")
        time.sleep(wait_time)

    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'URL': df['Links'],
        'Title': titles,
        'Content': contents,
    })

    # Write to an Excel file
    result_df.to_excel(output_file_path, index=False)
    print(f"Scraping completed and saved to {output_file_path}")

# Start the scraping process
scrape_from_excel(input_file_path, output_file_path)