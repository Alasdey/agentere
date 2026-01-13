import sys
import os
import json

# =============================================================================
# SETUP PATHS
# =============================================================================
# Add the parent directory to sys.path so we can import 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.dataprep import load_hf_dataset_parsed

def run_test():
    # Configuration
    # You may need to change these column names depending on the specific dataset's schema
    REPO_ID = "Nofing/Hievents-span" 
    SPLIT = "train"       # Using validation usually provides a good sample
    TEXT_FIELD = "text"        # The column containing the document
    ANN_FIELD = "annots"  # The column containing the <e1> label <e2> string
    MAX_EXAMPLES = 5
    
    print(f"=== Testing Custom Loader Logic ===")
    print(f"Dataset: {REPO_ID}")
    print(f"Split: {SPLIT}")
    print(f"Looking for annotations in column: '{ANN_FIELD}'")
    print(f"Regex Expectation: <ID CONTENT> LABEL <ID CONTENT>")
    print("="*60)

    try:
        # Initialize the generator
        loader = load_hf_dataset_parsed(
            repo_id=REPO_ID,
            split=SPLIT,
            text_field=TEXT_FIELD,
            ann_field=ANN_FIELD,
            max_examples=MAX_EXAMPLES,
            streaming=True # Use streaming to avoid downloading the whole thing for a test
        )

        for item in loader:
            print(f"\n[Document ID]: {item['id']}")
            print(f"[Language]:    {item['lang']}")
            
            # Print Text Snippet
            text_preview = item['doc_text'][:100].replace('\n', ' ') + "..." if len(item['doc_text']) > 100 else item['doc_text']
            print(f"[Text Snippet]: {text_preview}")
            
            # Print Extracted Triples
            triples = item['gold_triples']
            print(f"[Triples Found]: {len(triples)}")
            
            if triples:
                for i, (src, lbl, tgt) in enumerate(triples, 1):
                    print(f"  {i}. {src} --[{lbl}]--> {tgt}")
            else:
                print("  (No triples matched the Regex pattern)")
                print(f"  Raw Annotation Snippet (Verification):")
                # This helps debug if the regex is wrong or the column is empty
                # We can't access 'ann_text' directly from item, but we rely on the fact 
                # that if triples are 0, the parsing returned nothing.
                pass

            print("-" * 60)

    except Exception as e:
        print(f"\nERROR: Failed to load dataset.")
        print(f"Details: {e}")
        print("\nTroubleshooting Tips:")
        print("1. Check if 'text_field' and 'ann_field' match the column names in the actual dataset.")
        print("2. Ensure the annotation column contains strings, not JSON objects.")

if __name__ == "__main__":
    run_test()