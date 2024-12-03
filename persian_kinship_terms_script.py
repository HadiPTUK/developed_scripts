import farsnet

sense_service = farsnet.sense_service

def extract_terms_from_synset(synset_id, level=0, visited=set()):
    """
    Extract kinship terms recursively starting from a root synset.
    """
    if synset_id in visited:
        return []

    visited.add(synset_id)
    kinship_terms = []

    try:
        senses = sense_service.get_senses_by_synset(synset_id)
        for sense in senses:
            kinship_terms.append(sense.value)
            print("  " * level + f"- {sense.value}")

        relations = sense_service.get_sense_relations_by_id(synset_id)
        for relation in relations:
            if relation.type_ == "hyponym":
                kinship_terms += extract_terms_from_synset(relation.sense2_id(), level + 1, visited)
    except Exception as e:
        print(f"Error processing synset {synset_id}: {e}")

    return kinship_terms

def find_root_synset(keyword):
    """
    Finds the synset ID for a root concept like 'خانواده'.
    """
    try:
        senses = sense_service.get_senses_by_word("START", keyword)
        for sense in senses:
            print(f"Root Sense Found: {sense.value}, Synset ID: {sense.synset_id}")
            return sense.synset_id
    except Exception as e:
        print(f"Error finding root synset for {keyword}: {e}")
    return None

if __name__ == "__main__":
    print("Finding root synset for 'خانواده'...")
    root_synset_id = find_root_synset("خانواده")

    if root_synset_id:
        print(f"\nExtracting kinship terms starting from Synset ID: {root_synset_id}...\n")
        kinship_terms = extract_terms_from_synset(root_synset_id)

        unique_kinship_terms = sorted(set(kinship_terms))

        print("\nFinal Extracted Kinship Terms:")
        for term in unique_kinship_terms:
            print(term)
    else:
        print("Root synset for kinship not found.")
