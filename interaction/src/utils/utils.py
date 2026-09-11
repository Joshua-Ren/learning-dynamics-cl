
import torch
import re

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if solution is None:
        return 'N/A'
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def sharegpt_format(example):
    q = example["question"].strip()
    a = example["answer"].strip()
    instruction = 'Please answer the following question and give answer starting with "####".'
    output = instruction + "\nquestion: " + q + "\nanswer: " + a
    return {"text":output,
            "instruction": instruction + "\nquestion: " + q + "\nanswer: ",
            "ground_truth": extract_solution(a),
            "answer": a
        }
