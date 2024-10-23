import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import re
import time
import random

input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping data/Scrapping Input.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/Scrapping data/Scrapping Output.xlsx"

unwanted_phrases = list(set([
    "View more news", "opens new tab", "Our Standards:", "The graph shows the current", "This article is more than",
    "Click here to view the list", "Gift 5 articles", "Subscribe", "Follow the topics",
    "Login", "Already a subscriber?", "Subscribe for all of The Times", "View Report", "Fetching latest articles",
    "Disclaimer", "While we try everything to ensure accuracy",
    "The designations employed and the presentation of material on the map",
    "more taiwan news  2024 all rights reserved  ",
    "Copy link Copied Copy link Copied  to gift this article  to anyone you choose each month when you subscribe.",
    "(Reuters)",
]))

def clean_content(content):
    for phrase in unwanted_phrases:
        content = content.replace(phrase, "")

    content = re.sub(r'[^\w\s]', '', content)
    content = content.lower() 
    return content.strip()

def scrape_with_cloudscraper(url):
    attempts = 3
    for attempt in range(attempts):
        try:
            print(f"Attempting with CloudScraper for URL: {url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                title = scrape_title(soup)
                if not title:
                    print("Title not found!")
                    return None

                if "event summary" in title.lower():
                    print(f"Skipping URL: {url} because it contains 'Event summary'")
                    return None

                # Extract the content
                content = scrape_content(soup)
                if not content:
                    print("Content not found!")
                    return None

                cleaned_content = clean_content(content)

                cleaned_title = clean_content(title)

                return {"title": cleaned_title, "content": cleaned_content}
            else:
                print(f"CloudScraper failed with status code: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error with CloudScraper: {e}")
            if attempt < attempts - 1:
                wait_time = random.uniform(1, 3)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    return None

def scrape_title(soup):
    """Try multiple patterns to extract the title."""
    title_patterns = [
        {'tag': 'h1', 'attr': {'class': '_804759db85ec8cc17e4f', 'data-testid': 'ArticleHeader-headline'}},
        {'tag': 'div', 'attr': {'id': 'alert_TC_Green_title_big'}},
        {'tag': 'span', 'attr': {'id': 'ctl00_masterReportTitle'}},
        {'tag': 'h1'},
        {'tag': 'h2'},
        {'tag': 'div', 'attr': {'class': 'view_headline LoraMedium'}},
        {'tag': 'div', 'attr': {'data-qa': 'headline'}},
        {'tag': 'h1', 'attr': {'class': 'css-xyz'}},
        {'tag': 'div', 'attr': {'class': 'alert_title'}},
    ]

    for pattern in title_patterns:
        tag = pattern.get('tag')
        attr = pattern.get('attr', {})
        title_element = soup.find(tag, attr)
        if title_element:
            return title_element.text.strip()

    return None

def scrape_content(soup):
    content_patterns = [
        {'tag': 'div', 'attr': {'data-testid': lambda x: x and x.startswith('paragraph-')}},
        {'tag': 'p'},
        {'tag': 'td', 'attr': {'class': 'cell_value_summary'}},
        {'tag': 'span', 'attr': {'class': 'read'}},
        {'tag': 'p', 'attr': {'class': 'p_summary'}},
        {'tag': 'p', 'attr': {'class': 'css-at9mc1 evys1bk0'}},
    ]

    for pattern in content_patterns:
        tag = pattern.get('tag')
        attr = pattern.get('attr', {})
        elements = soup.find_all(tag, attr)
        if elements:
            content = " ".join([el.text.strip() for el in elements])
            return content

    return None

def scrape_from_excel(input_file_path, output_file_path):
    df = pd.read_excel(input_file_path)

    titles, contents, locations = [], [], []

    for index, row in df.iterrows():
        url = row['Links']
        print(f"Scraping URL {index + 1}: {url}")

        result = scrape_with_cloudscraper(url)
        if result:
            titles.append(result['title'])
            contents.append(result['content'])
        else:
            titles.append("")
            contents.append("")

        wait_time = random.uniform(1, 5)
        print(f"Waiting for {wait_time:.2f} seconds before the next request...")
        time.sleep(wait_time)

    result_df = pd.DataFrame({
        'URL': df['Links'],
        'Title': titles,
        'Content': contents,
    })

    result_df.to_excel(output_file_path, index=False)
    print(f"Scraping completed and saved to {output_file_path}")

scrape_from_excel(input_file_path, output_file_path)
