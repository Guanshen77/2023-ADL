import json
import jsonlines
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="get command")
    parser.add_argument(
        "--jsonl_file",
        type=str,
        default=None,
        help="input",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=None,
        help="output",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Read the JSONL file
    with open(args.jsonl_file, "r") as jsonl_file:
        # Initialize an empty list to store JSON objects
        json_data = []
        for line in jsonl_file:
            # Parse each line as a JSON object
            data = json.loads(line)
            json_data.append(data)
    # Write the JSON data to a regular JSON file
    with open(args.json_file, "w") as json_file:
        json.dump(json_data, json_file, indent=2)

    print("get json file")

if __name__ == "__main__":
    main()
