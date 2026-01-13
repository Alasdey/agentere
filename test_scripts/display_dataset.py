
from datasets import load_dataset

def load_first_row(dataset_id):
    try:
        # Load the dataset from Hugging Face
        print(f"Loading dataset: {dataset_id}...")
        dataset_dict = load_dataset(dataset_id)

        # Print available splits (e.g., train, test, validation)
        available_splits = list(dataset_dict.keys())
        print(f"Available splits: {available_splits}")

        # Select the first split found (usually 'train')
        # You can manually specify a split like dataset_dict['train'] if you prefer
        first_split_name = available_splits[0]
        selected_split = dataset_dict[first_split_name]

        # Access the first row (index 0)
        first_row = selected_split[0]

        print(f"\n--- First Row from '{first_split_name}' split ---")
        
        # Print nicely formatted if it's a dictionary
        if isinstance(first_row, dict):
            for key, value in first_row.items():
                print(f"{key}: {value}")
        else:
            print(first_row)
            
        return first_row

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # The specific dataset you requested
    dataset_name = "Nofing/EventStoryLine-1.5-span"
    
    load_first_row(dataset_name)
