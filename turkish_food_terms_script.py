from WordNet.WordNet import WordNet
from WordNet.SemanticRelation import SemanticRelation
from WordNet.InterlingualRelation import InterlingualRelation

def find_relations_for_word(word, domain):
    """
    Finds the SynSets and their Hypernym and Hyponym relations for the given word.
    Prints the details of each SynSet and its relations.
    """
    print(f"\nSynSets and relations for the word '{word}':\n")
    # Get all SynSets associated with the word
    synsets = domain.getSynSetsWithLiteral(word)
    if synsets:
        # Display each SynSet and its relations
        for synset in synsets:
            display_synset_and_relations(synset, domain)
    else:
        # Notify if no SynSet is found for the word
        print(f"No SynSet found for the word '{word}'.")

def find_relations_for_id(synset_id, domain):
    """
    Finds the SynSet and its relations for the given SynSet ID.
    Prints the details of the SynSet and its relations.
    """
    print(f"\nSynSet and relations for the ID '{synset_id}':\n")
    # Get the SynSet associated with the ID
    synset = domain.getSynSetWithId(synset_id)
    if synset:
        # Display the SynSet and its relations
        display_synset_and_relations(synset, domain)
    else:
        # Notify if no SynSet is found for the ID
        print(f"No SynSet found for the ID '{synset_id}'.")

def display_synset_and_relations(synset, domain):
    """
    Displays details of a SynSet, including its literals, definition, and relations.
    """
    # Print SynSet ID
    print(f"\nSynSet ID: {synset.getId()}")
    # Retrieve and display all literals (words) in the SynSet
    literals = [synset.getSynonym().getLiteral(i).getName() for i in range(synset.getSynonym().literalSize())]
    print(f"Words: {', '.join(literals)}")
    # Print the definition of the SynSet
    print(f"Definition: {synset.getLongDefinition()}")
    # Display Hypernyms of the SynSet
    print("\nHypernyms:")
    display_relations(synset, domain, relation_type="Hypernym")
    # Display Hyponyms of the SynSet
    print("\nHyponyms:")
    display_relations(synset, domain, relation_type="Hyponym")

def display_relations(synset, domain, relation_type="Hypernym"):
    """
    Displays the specified type of relations (Hypernym or Hyponym) for the given SynSet.
    Prints related SynSet IDs and their definitions if available.
    """
    for i in range(synset.relationSize()):
        # Retrieve each relation for the SynSet
        relation = synset.getRelation(i)
        # Check if the relation is a SemanticRelation and matches the requested type
        if isinstance(relation, SemanticRelation) and relation.getRelationType().name == relation_type.upper():
            # Find the related SynSet by its ID
            related_synset = domain.getSynSetWithId(relation.getName())
            if related_synset:
                # Print the related SynSet's ID and definition
                print(f"- {related_synset.getId()}: {related_synset.getLongDefinition()}")
            else:
                # Notify if the related SynSet's definition is not found
                print(f"- {relation.getName()} (Definition not found)")
        # If the relation is an InterlingualRelation, display its ID
        elif isinstance(relation, InterlingualRelation):
            print(f"- InterlingualRelation: {relation.getName()} (Interlingual relation)")

def main():
    """
    Main function to load the WordNet file and allow the user to search
    for SynSets and their relations by word or SynSet ID.
    """
    # Specify the path to the WordNet XML file
    domain_file = "WordNet/data/turkish_wordnet.xml"
    # Load the WordNet data from the file
    domain = WordNet(domain_file)
    print("WordNet file loaded successfully.")

    while True:
        # Prompt the user to choose the type of search
        search_type = input("\nChoose search type ('word' for a word, 'id' for an ID, 'exit' to quit): ").strip().lower()
        if search_type == "exit":
            # Exit the program
            print("Exiting...")
            break
        elif search_type == "word":
            # Prompt the user to enter a word for search
            word = input("\nEnter a word: ").strip().lower()
            find_relations_for_word(word, domain)
        elif search_type == "id":
            # Prompt the user to enter a SynSet ID for search
            synset_id = input("\nEnter a SynSet ID: ").strip()
            find_relations_for_id(synset_id, domain)
        else:
            # Notify the user of invalid input
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
