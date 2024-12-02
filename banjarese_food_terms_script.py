import xml.etree.ElementTree as ET
import re
import mwparserfromhell

from datasets import load_dataset
import spacy
from spacy.tokenizer import Tokenizer
from gensim.models import Word2Vec
import numpy as np
import re, unicodedata

import csv

def parse_bjn_wiktionary(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    # Extract pages, excluding templates and the main page
    pages = [page for page in root if "page" in page.tag and ":" not in page[0].text]
    pages = pages[1:]  # Skip the main page

    # Pattern to identify entries relevant to "Banjar"
    banjar_pattern = r'{{[\W]*[Bb]anjar[\W]*}}'

    # Helper function: Check if a whole word exists in a string
    def is_whole_word_within_string(main_string, sub_string):
        pattern = rf'\b{re.escape(sub_string)}\b'
        return bool(re.search(pattern, main_string))

    # Exclude certain parts of speech
    exclude_pos = [
        "kata sipat", "kata sifat", "kata katarangan", "kata banda",
        "kata gawi", "kata hubung", "kata kerja", "kata panghubung"
    ]

    # Parse terms and rough definitions from pages
    terms_and_rough_definitions = []
    for page in pages:
        term = page[0].text.lower()
        revision = next(elem for elem in page if "revision" in elem.tag)
        text = next(elem for elem in revision if "text" in elem.tag).text

        # Skip pages not containing the "Banjar" pattern
        if not re.search(banjar_pattern, text):
            continue

        terms_and_rough_definitions.append((term, text))

    # Process terms and definitions
    terms_and_definitions = []
    for term, raw_text in terms_and_rough_definitions:
        parsed = mwparserfromhell.parse(raw_text)

        # Extract and clean the main content section
        sections = parsed.get_sections(levels=[2])
        stripped_text = sections[0].strip_code().lower().strip() if sections else ""

        # Clean unnecessary whitespace
        for pattern, replacement in [
            (" \n", "\n"), ("\n ", "\n"), (":\n", ": "), ("\nkata", "\n\nkata"), ("\n\n\n", "\n\n")
        ]:
            stripped_text = stripped_text.replace(pattern, replacement)

        # Extract and filter definitions
        processed_text = stripped_text.split("\n\n")[0]
        definitions = [
            d for d in processed_text.split("\n")
            if not is_whole_word_within_string(d, term) and "tumbung:" not in d
        ]

        # Handle fallback cases for empty or alternative spellings
        if not definitions or definitions[0] == "ejaan lain":
            alt_section_index = 1 if definitions else 0
            processed_text = stripped_text.split("\n\n")[alt_section_index]
            definitions = [
                d for d in processed_text.split("\n")
                if d not in exclude_pos and not is_whole_word_within_string(d, term)
            ]

        # Final fallback for empty definitions
        if not definitions:
            definitions = [term]

        terms_and_definitions.append((term, definitions))

    # Flatten the results for easier access
    terms_and_definitions_cleaned = [
        [term, definition]
        for term, definitions in terms_and_definitions
        for definition in definitions
    ]

    # Output: terms_and_definitions now contains (title, definition) pairs
    return terms_and_definitions_cleaned

def parse_bjn_wikipedia(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    # Helper function to filter valid pages
    def filter_pages(pages):
        filtered_pages = [
            page for page in pages
            if "page" in page.tag
            and ":" not in page[0].text  # Exclude templates
            and not page[0].text.lower().startswith("daptar ")  # Exclude list articles
            and not page[0].text.lower().isnumeric()  # Exclude year articles
        ]
        return filtered_pages[1:]  # Exclude main page

    # Filter and process pages
    pages = filter_pages(list(root))

    # Extract terms and definitions
    terms_and_definitions = []
    for page in pages:
        title = page[0].text.lower()
        revision = next(elem for elem in page if "revision" in elem.tag)
        text = next(elem for elem in revision if "text" in elem.tag).text

        # Find and parse the first relevant line containing "adalah"
        for line in text.split("\n"):
            if "adalah" in line and "convert |" not in line:
                parsed_line = mwparserfromhell.parse(line).strip_code().lower()
                terms_and_definitions.append([title, parsed_line])
                break  # Stop after the first match

    # Output: terms_and_definitions now contains (title, definition) pairs
    return terms_and_definitions

def calculate_similarities(terms_and_definitions):
    def clean_non_alphanumeric(text):
        """Normalize and clean text by removing non-alphanumeric characters."""
        normalized_text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        cleaned_text = re.sub(r'[^a-zA-Z0-9]+', ' ', normalized_text).strip()
        return cleaned_text

    def compute_cosine_similarities(main_vector, other_vectors):
        """Compute cosine similarities between a main vector and multiple other vectors."""
        main_norm = np.linalg.norm(main_vector)
        other_norms = np.linalg.norm(other_vectors, axis=1)
        valid_indices = (main_norm != 0) & (other_norms != 0)  # Avoid division by zero
        similarities = np.zeros(len(other_vectors))
        if valid_indices.any():
            similarities[valid_indices] = (
                np.dot(other_vectors[valid_indices], main_vector) /
                (main_norm * other_norms[valid_indices])
            )
        return similarities
    
    # Load dataset and combine data sources
    dataset = load_dataset("acul3/KoPI-NLLB", "bjn_Latn-neardup")
    sentences = dataset["train"]["text"]

    terms_and_defs_combined = [" ".join(word_def) for word_def in terms_and_definitions]
    all_text_data = sentences + terms_and_defs_combined

    # Load SpaCy tokenizer
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)

    # Tokenize text data
    tokens = [
        [token.text.lower() for token in tokenizer(clean_non_alphanumeric(text))]
        for text in all_text_data
    ]

    # Train Word2Vec model
    word2vec_model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, sg=0)

    # Process the main sentence
    main_sentence = "makanan adalah samunyaan nang kawa dimakan macam nasi iwak wadai"
    main_tokens = [token for token in main_sentence.split(" ") if token in word2vec_model.wv]
    main_vector = np.mean([word2vec_model.wv[token] for token in main_tokens], axis=0)

    # Vocabulary from the Word2Vec model
    vocab = set(word2vec_model.wv.index_to_key)

    # Prepare word-definition vectors in bulk
    word_def_tokens = [
        [token for token in " ".join([word, definition]).split(" ") if token in vocab]
        for word, definition in terms_and_definitions
    ]

    # Filter out empty token lists
    valid_word_defs = [
        (word, definition, tokens) for (word, definition), tokens in zip(terms_and_definitions, word_def_tokens) if tokens
    ]

    # Convert tokens to vectors
    word_def_vectors = np.array([
        np.mean([word2vec_model.wv[token] for token in tokens], axis=0) for _, _, tokens in valid_word_defs
    ])

    # Compute similarities in bulk
    similarities = compute_cosine_similarities(main_vector, word_def_vectors)

    # Combine results
    word_definition_similarities = [
        (word, definition, sim) for (word, definition, _), sim in zip(valid_word_defs, similarities)
    ]

    return word_definition_similarities

wiktionary_parsed = parse_bjn_wiktionary("bjnwiktionary-20231201-pages-articles.xml")
wikipedia_parsed  = parse_bjn_wikipedia("bjnwiki-20231220-pages-articles.xml")
all_parsed = wiktionary_parsed + wikipedia_parsed
terms_defs_similarities = calculate_similarities(all_parsed)

with open("bjn_terms_defs_similarities.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(terms_defs_similarities)
