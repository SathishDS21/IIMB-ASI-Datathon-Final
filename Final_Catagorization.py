import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from collections import Counter

# Define input and output file paths
input_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM/Final Output2.xlsx"
output_file_path = "/Users/sathishm/Documents/TSM Folder/Datathon Stage 2/LLM/LLM Output.xlsx"

# Load the pretrained BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Define keyword categories
category_keywords = {
    "Trade": [
        "trade", "economy", "import", "export", "market", "tariff", "commodity", "goods", "supply chain", "trade war",
        "free trade", "trade agreement", "negotiations", "quota", "price index", "merchandise", "globalization",
        "wholesale", "retail", "demand", "surplus", "deficit", "production", "trade barrier", "customs", "tax",
        "exemption", "embargo", "foreign exchange", "investment", "subsidy", "licensing", "capital", "revenue",
        "income", "corporation", "bonds", "securities", "shareholders", "stocks", "transactions", "commerce",
        "futures", "commodities", "purchasing", "economic sanctions", "economic impact", "trade deal", "valuation",
        "global trade", "trade unions", "business", "supply curve", "international market", "balance of trade",
        "tariff barriers", "economic recovery", "commercial agreement", "commodity prices", "financial markets",
        "global trade patterns", "globalization index", "foreign trade policies", "trade credit", "trade partnerships",
        "exchange rates", "tariff imposition", "trade imbalances", "export surplus", "international commerce",
        "consumer goods", "multilateral agreements", "economic development", "gross domestic product",
        "retail industry",
        "stock exchange", "economic policy", "venture capital", "financial transactions", "revenue generation",
        "economic stability", "cross-border trade", "trading platforms", "corporate earnings", "business expansion",
        "economic indicators", "trade reform", "consumer demand", "economic growth", "inflation rate",
        "commodity trade",
        "market competition", "industrial production", "international investment", "financial assets", "capital flow"
    ],
    "Natural Disaster": [
        "hurricane", "earthquake", "flood", "storm", "disaster", "cyclone", "wildfire", "tsunami", "avalanche",
        "volcano", "tornado", "drought", "landslide", "eruption", "heatwave", "monsoon", "blizzard", "hailstorm",
        "lightning", "snowstorm", "tremor", "aftershock", "seismic", "natural hazard", "climate", "global warming",
        "flash flood", "forest fire", "tropical storm", "typhoon", "severe weather", "weather alert", "disaster relief",
        "damage", "casualties", "rescue", "evacuation", "tsunami warning", "risk assessment", "disaster preparedness",
        "recovery efforts", "humanitarian", "epicenter", "famine", "catastrophe", "flooding", "mudslide",
        "cyclonic storm",
        "environmental disaster", "storm surge", "tectonic plate", "seismic waves", "geological event", "lava flow",
        "ash cloud", "environmental impact", "natural catastrophe", "extreme heat", "disaster response",
        "reconstruction",
        "earthquake tremors", "volcanic eruption", "glacial melting", "tidal waves", "windstorm", "severe flooding",
        "hazardous weather", "severe drought", "floodwaters", "storm damage", "structural damage", "rescue operations",
        "rebuilding efforts", "natural calamity", "mass evacuation", "emergency response", "environmental hazards",
        "heat index", "intense rainfall", "climate disaster", "wildfire spread", "debris flow", "ecosystem destruction",
        "geological shift", "lava eruption", "natural relief", "reconstruction phase", "rapid onset disaster",
        "storm path",
        "weather disaster", "natural hazards", "torrential rain", "ocean storm"
    ],
    "Geopolitics": [
        "politics", "war", "sanction", "geopolitical", "conflict", "diplomacy", "tension", "peace talks", "alliance",
        "rebellion", "treaty", "territory", "sovereignty", "military", "occupation", "border", "regime", "coup",
        "nuclear", "weapon", "embassy", "ambassador", "intervention", "foreign relations", "defense", "security",
        "spy", "terrorism", "proxy war", "insurgency", "diplomatic", "international law", "revolution", "espionage",
        "invasion", "sanctions", "ceasefire", "blockade", "peacekeeping", "human rights", "arms deal", "intelligence",
        "civil war", "uprising", "authoritarian", "dictatorship", "power struggle", "sovereignty", "statecraft",
        "international conflicts", "strategic interests", "military presence", "border disputes", "conflict resolution",
        "military strategy", "geopolitical landscape", "foreign intervention", "political unrest", "arms control",
        "strategic resources", "international stability", "geostrategy", "intelligence operations",
        "international borders",
        "economic sanctions", "state sovereignty", "political power", "foreign policy", "embargo enforcement",
        "government leadership", "global governance", "international relations", "power balance", "allied forces",
        "military coup", "global influence", "nation-state", "diplomatic relations", "territorial integrity",
        "sovereign state", "political authority", "territorial claims", "cross-border conflict", "unilateral actions",
        "military enforcement", "foreign affairs", "diplomatic channels", "state-sponsored", "political instability",
        "geopolitical tension", "security council", "military occupation", "international influence",
        "peace negotiations"
    ],
    "Transportation": [
        "transport", "logistics", "shipping", "freight", "airline", "port", "cargo", "rail", "airport", "shipping lane",
        "container", "trucks", "railway", "highway", "routes", "infrastructure", "customs clearance", "warehousing",
        "inventory", "passenger", "bus", "transit", "cruise", "flight", "pilot", "vessel", "delivery",
        "package", "carriage", "freight train", "logistics hub", "freight forwarder", "distribution", "sea freight",
        "air freight", "ocean freight", "road transportation", "shipment", "maritime", "pipeline", "logistics company",
        "drone delivery", "border crossing", "rail cargo", "trucking", "automobile", "transit system",
        "shipping logistics",
        "logistics services", "freight forwarding", "supply lines", "port operations", "road network",
        "aviation industry",
        "public transportation", "urban transit", "passenger transport", "maritime transport", "rail networks",
        "freight logistics", "cargo shipments", "supply routes", "shipping costs", "delivery networks", "air cargo",
        "cargo handling", "freight movement", "transportation infrastructure", "transport hub", "container ship",
        "rail freight", "transportation services", "port facilities", "rail cargo transport", "bulk freight",
        "transport fleet",
        "customs logistics", "road freight", "freight volume", "transit authority", "vehicle transportation",
        "global transport",
        "shipment logistics", "land transport", "multimodal logistics", "transportation route", "inland transport"
    ],
    "Others": [
        "incident", "event", "unknown", "miscellaneous", "general", "unspecified", "undefined", "situation", "case",
        "unknown reason", "unsure", "various factors", "multiple causes", "different issues", "varied",
        "general category", "other", "not listed", "miscellaneous reason", "unknown cause", "not specified",
        "unknown situation", "general impact", "other reasons", "uncertain impact", "unclear event", "unrelated factor",
        "broad issue", "wide-ranging", "unclassified", "broad category", "ambiguous situation", "various scenarios",
        "mixed circumstances", "indeterminate", "unknown origin", "general issue", "uncertain", "ambiguous",
        "general impact",
        "unclear scenario", "assorted", "undefined cause", "vague reason", "diverse reasons", "unclassified",
        "broad impact",
        "unfamiliar", "nonspecific", "non-categorized", "non-specific issue", "general event", "unsure situation",
        "unexplained event", "general impact", "undetermined outcome", "uncertain scenario", "miscellaneous factors",
        "not specifically stated", "broad spectrum", "vague event", "unclear category", "inconclusive",
        "non-determined",
        "unresolved issue", "ambiguous context", "multifactorial", "unconnected factors", "diverse causes",
        "generalized impact",
        "non-classified", "unclear reasoning", "mixed results", "uncertain origin", "general topic", "uncertain case", "wide scope",
        "unknown implications", "vague circumstances", "generalized outcome", "multiple possibilities",
        "uncertain outcome",
        "unresolved situation", "multiple reasons", "unknown factors", "non-defined", "broad reasons",
        "ambiguous reasons",
        "uncertain explanation", "miscellaneous occurrence", "non-categorized event", "unsure outcome",
        "unresolved cause",
        "varied implications", "unfamiliar situation", "unclear status", "open-ended", "vague incident", "nondescript"
    ]
}

def clean_text(text):
    """Clean the text by removing unwanted characters."""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def categorize_text(text):
    """Categorize text based on the most repeated keywords from each category."""
    text = clean_text(text)

    keyword_count = Counter()

    # Count occurrences of keywords in each category
    for category, keywords in category_keywords.items():
        keyword_count[category] = sum(keyword in text for keyword in keywords)

    # Get the category with the highest count of matched keywords
    most_common_category = keyword_count.most_common(1)[0][0] if keyword_count else "Others"

    return most_common_category

# Read input Excel file
df = pd.read_excel(input_file_path)

# Process each row and categorize content based on keyword counts
df['Category'] = df['Content'].apply(categorize_text)

# Save output to new Excel file
df.to_excel(output_file_path, index=False)

print(f"Categorization completed and saved to {output_file_path}")