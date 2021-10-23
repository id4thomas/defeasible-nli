import json

def read_jsonl_inputs(input_file):
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def read_lines(input_file):
    with open(input_file, "r") as f:
        lines=f.read().split("\n")
    return lines