
from datasets import load_dataset

def load_first_row(dataset_id, number_of_rows):
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

        for i in range(number_of_rows):
            # Access the first row (index 0)
            row = selected_split[i]

            print(f"\n--- Row {i} from '{first_split_name}' split ---")
            
            # Print nicely formatted if it's a dictionary
            if isinstance(row, dict):
                for key, value in row.items():
                    print(f"{key}: {value}")
            else:
                print(row)
            
        return

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # The specific dataset you requested
    # dataset_name = "Nofing/MAVEN-ERE-Causal-Events"
    dataset_name = "Nofing/EventStoryLine-1.5-Causal"
    number_of_rows = 5    
    load_first_row(dataset_name, number_of_rows)
