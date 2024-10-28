import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import re
import time
import random
import openai
from sklearn.metrics import accuracy_score
from tqdm import tqdm

openai.api_key = "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"

input_file_path = "D:/Python/Scrapping Input.xlsx"
output_file_path = "D:/Python/Combined_Output.xlsx"
true_category_file_path = "D:/Python/True Cat.xlsx"

categories = ["Natural Disaster", "Geo-politics", "Trade", "Labor", "Transportation", "Others"]

unwanted_phrases = list(set([
    "View more news", "opens new tab", "Our Standards:", "The graph shows the current", "This article is more than",
    "Click here to view the list", "Gift 5 articles", "Subscribe", "Follow the topics", "Login",
    "Already a subscriber?", "Subscribe for all of The Times", "View Report", "Fetching latest articles",
    "Disclaimer", "While we try everything to ensure accuracy",
    "The designations employed and the presentation of material on the map",
    "more taiwan news  2024 all rights reserved",
    "Copy link Copied Copy link Copied to gift this article to anyone you choose each month when you subscribe.",
    "(Reuters)"
]))

def clean_content(content):
    for phrase in unwanted_phrases:
        content = content.replace(phrase, "")
    content = re.sub(r'[^\w\s,.!?\'"]+', '', content)
    return content.strip()

def call_openai_api(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,  # Adjust temperature for better variety
        frequency_penalty=0.3  # Reduced penalty for better response diversity
    )
    return response['choices'][0]['text'].strip()

def summarize_content(content):
    prompt = f"Summarize the following news article in a single paragraph with more than 100 words:\n\n{content}"
    return call_openai_api(prompt)

def extract_relevant_location(content):
    prompt = (
        f"Identify only the most specific and relevant geopolitical location from the following content. "
        f"Provide the location as a state or country name only. If a city is mentioned, replace it with the corresponding state and country. "
        f"For example, if 'Los Angeles' is mentioned, respond with 'California, United States'. If 'Hong Kong' is mentioned, respond with 'Hong Kong, China'. "
        f"Do not include city names on their own.\n\n"
        f"Content: {content}\n\n"
        f"Respond with only the location name in the format 'State, Country' or 'Country' if only the country is mentioned."
    )
    response = call_openai_api(prompt)
    return response.strip()

def generate_category(content):
    prompt = (
        f"Classify the following content into one of the categories: "
        f"Transport (content that specifically discusses the physical movement of goods or people via pipelines, shipping, air, rail, or road, or transport infrastructure), "
        f"Natural Disaster (content about natural events like earthquakes, taks about impact of Tropical Cyclone, floods, hurricane, wind speed or wildfires, or the impact of such events on people or infrastructure or storm warning), "
        f"Geo-politics (content focusing on international relations, military conflicts, defense policies, and diplomatic negotiations and not hurricanes), "
        f"Trade (content involving business transactions, production, imports, exports, tariffs, trade agreements, stock market activity, corporate acquisitions, or economic sanctions, trade regulations, trade restrictions), "
        f"Labour (content that specifically discusses workers, two-tier wage systems, strike risks, union strikes, labor unions, wage negotiations, layoffs, or changes in employment conditions), "
        f"Others (use this category for any content not related to the above categories).\n\n"
        f"Content: {content}\n"
        f"Respond with only the category name."
    )
    return call_openai_api(prompt)

def predict_supply_chain_impact(content, category):
    prompt = (
        f"Based on the category '{category}' and the following content, identify the single most relevant supply chain area impacted. "
        f"Choose from these areas: \n"
        f"1. Procurement: Sourcing and acquiring materials or goods required for production. Note that tariffs or trade restrictions can increase costs and disrupt sourcing efforts.\n"
        f"2. Production: Manufacturing or assembling products, including labor and machinery management.\n"
        f"3. Logistics: Movement and storage of goods within and outside the organization, or physical movement of goods to customers, including all shipping modes.\n"
        f"4. Inventory Management: Monitoring and controlling stock levels to meet demand.\n"
        f"5. Demand Planning and Forecasting: Predicting future demand to align production and inventory with market needs.\n"
        f"6. Customer Service and Returns Management (Reverse Logistics): Handling customer inquiries and managing returns or exchanges.\n"
        f"7. Supplier Management: Managing supplier relationships to ensure quality, timely delivery, and cost efficiency. Supplier Management is especially important in situations involving collaborative projects, trade restrictions, or the need for sourcing alternatives.\n"
        f"8. Risk Management: Identifying and mitigating risks across the supply chain, especially those related to geopolitical changes like tariffs or trade restrictions.\n\n"
        f"Identify the primary event or information affecting the supply chain in the content and focus on the most relevant area impacted. "
        f"For instance:\n"
        f"- Trade restrictions may impact Procurement or Supplier Management if they affect sourcing materials from specific countries.\n"
        f"- Collaborative projects involving multiple organizations may require strong Supplier Management to coordinate sourcing between parties.\n"
        f"- Tariffs on imports can increase costs in Procurement, leading to a need for alternative sourcing options.\n\n"
        f"If there are multiple unrelated events, select only the one that most likely impacts a specific supply chain area.\n\n"
        f"Content: {content}\n\n"
        f"Respond in the format 'supply_chain_impact: <area>, reason_for_impact: <reason>'"
    )
    response = call_openai_api(prompt)

    # Extract supply chain impact and reason using regular expressions
    impact_match = re.search(r'supply_chain_impact:\s*(.*?)(?:,|$)', response)
    reason_match = re.search(r'reason_for_impact:\s*(.*)', response)

    supply_chain_impact = impact_match.group(1).strip() if impact_match else "Unclear Impact"
    reason_for_impact = reason_match.group(1).strip() if reason_match else "The impact on supply chain is not specified."

    return supply_chain_impact, reason_for_impact

    response = call_openai_api(prompt)

    # Extract supply chain impact and reason using regular expressions
    impact_match = re.search(r'supply_chain_impact:\s*(.*?)(?:,|$)', response)
    reason_match = re.search(r'reason_for_impact:\s*(.*)', response)

    supply_chain_impact = impact_match.group(1).strip() if impact_match else "Unclear Impact"
    reason_for_impact = reason_match.group(1).strip() if reason_match else "The impact on supply chain is not specified."

    return supply_chain_impact, reason_for_impact
    response = call_openai_api(prompt)

    # Extract supply chain impact and reason using regular expressions
    impact_match = re.search(r'supply_chain_impact:\s*(.*?)(?:,|$)', response)
    reason_match = re.search(r'reason_for_impact:\s*(.*)', response)

    supply_chain_impact = impact_match.group(1).strip() if impact_match else "Unclear Impact"
    reason_for_impact = reason_match.group(1).strip() if reason_match else "The impact on supply chain is not specified."

    return supply_chain_impact, reason_for_impact

    response = call_openai_api(prompt)

    # Extract supply chain impact and reason using regular expressions
    impact_match = re.search(r'supply_chain_impact:\s*(.*?)(?:,|$)', response)
    reason_match = re.search(r'reason_for_impact:\s*(.*)', response)

    supply_chain_impact = impact_match.group(1).strip() if impact_match else "Unclear Impact"
    reason_for_impact = reason_match.group(1).strip() if reason_match else "The impact on supply chain is not specified."

    return supply_chain_impact, reason_for_impact

def scrape_with_cloudscraper(url):
    attempts = 3
    for attempt in range(attempts):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = scrape_title(soup)
                content = scrape_content(soup)
                if title and content:
                    cleaned_content = clean_content(content)
                    cleaned_title = clean_content(title)
                    summary = summarize_content(cleaned_content)
                    location = extract_relevant_location(cleaned_content)
                    category = generate_category(summary)
                    supply_chain_impact, reason = predict_supply_chain_impact(summary, category)
                    return {
                        "url": url,
                        "title": cleaned_title,
                        "content": summary,
                        "location": location,
                        "category": category,
                        "supply_chain_impact": supply_chain_impact,
                        "reason_for_impact": reason
                    }
        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(random.uniform(1, 3))
    return None

def scrape_title(soup):
    title_patterns = [{'tag': 'h1'}, {'tag': 'h2'}, {'tag': 'div', 'attr': {'class': 'view_headline LoraMedium'}}]
    for pattern in title_patterns:
        tag, attr = pattern.get('tag'), pattern.get('attr', {})
        title_element = soup.find(tag, attr)
        if title_element:
            return title_element.text.strip()
    return None

def scrape_content(soup):
    content_patterns = [{'tag': 'div', 'attr': {'data-testid': lambda x: x and x.startswith('paragraph-')}}, {'tag': 'p'}]
    for pattern in content_patterns:
        tag, attr = pattern.get('tag'), pattern.get('attr', {})
        elements = soup.find_all(tag, attr)
        if elements:
            return " ".join([el.text.strip() for el in elements])
    return None

def scrape_from_excel(input_file_path, output_file_path, true_category_file_path):
    df = pd.read_excel(input_file_path)
    true_category_df = pd.read_excel(true_category_file_path)

    results = []
    accuracies = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing URLs"):
        url = row['Links']
        start_time = time.time()
        result = scrape_with_cloudscraper(url)
        end_time = time.time()

        if result:
            true_category = true_category_df.loc[index, 'Category']
            result.update({'URL': url})

            link_accuracy = 100 if result['category'].lower() == true_category.lower() else 0
            accuracies.append(link_accuracy)
            results.append(result)

            print(f"Time taken: {end_time - start_time:.2f} seconds | Accuracy: {link_accuracy}%")

    overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    print(f"Overall Category Classification Accuracy: {overall_accuracy:.2f}%")

    result_df = pd.DataFrame(results,
                             columns=["url", "title", "content", "location", "category", "supply_chain_impact", "reason_for_impact"])
    result_df.to_excel(output_file_path, index=False)
    print(f"Scraping and processing completed. Results saved to {output_file_path}")

scrape_from_excel(input_file_path, output_file_path, true_category_file_path)
