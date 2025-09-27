import os

import datasets


def make_map_fn(split, source=None):
        def process_fn(example, idx):
            if source is None:
                data_source = example.pop("source")
            else:
                data_source = source
            question = example.pop("prompt")
            solution = example.pop("answer")
            
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": f"{data_source}-{idx}",
                },
            }
            return data

        return process_fn

if __name__ == '__main__':

    # Define all datasets to be processed
    data_sources = ['GPQA-TTT', 'AIME-TTT', 'AIME25', 'AMC-TTT', 'MATH-TTT']
    
    for data_source in data_sources:
        print(f"Processing dataset: {data_source}")
        
        # Check if dataset directory exists
        if not os.path.exists(data_source):
            print(f"Warning: Dataset directory {data_source} does not exist, skipping")
            continue
            
        # Check if train and test files exist
        train_file = os.path.join(data_source, 'train.json')
        test_file = os.path.join(data_source, 'test.json')
        
        if not os.path.exists(train_file):
            print(f"Warning: Train file {train_file} does not exist, skipping")
            continue
            
        if not os.path.exists(test_file):
            print(f"Warning: Test file {test_file} does not exist, skipping")
            continue

        try:
            train_dataset = datasets.load_dataset("json", data_files=train_file, split='train')
            test_dataset = datasets.load_dataset("json", data_files=test_file, split='train')

            train_dataset = train_dataset.map(function=make_map_fn("train", data_source), with_indices=True)
            test_dataset = test_dataset.map(function=make_map_fn("test", data_source), with_indices=True)

            train_dataset.to_parquet(os.path.join(data_source, 'train-simplerl.parquet'))
            test_dataset.to_parquet(os.path.join(data_source, 'test-simplerl.parquet'))
            
            print(f"Successfully processed dataset: {data_source}")
            
        except Exception as e:
            print(f"Error processing dataset {data_source}: {e}")
            continue
