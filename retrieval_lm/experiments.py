import json
import os
from utils import load_jsonlines
from run_short_form import preprocess_input_data

def main():
    eval0_2_path = "../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_trs0-2-35.json"
    eval0_5_path = "../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_trs0-5-35.json"
    # eval0_8_path = "../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_trs0-8.json"
    # eval_path = "../../../../dataspace/P76124574/SELF-RAG/eval_data/popqa_longtail_w_gs.jsonl"
    
    eval0_2 = json.load(open(eval0_2_path))
    eval0_5 = json.load(open(eval0_5_path))
    # eval0_8 = json.load(open(eval0_8_path))
    # eval = load_jsonlines(eval_path)
    
    # print("0.2 metric_mean: ", eval0_2["metric_mean"])
    # print("0.5 metric_mean: ", eval0_5["metric_mean"])
    # print("0.8 metric_mean: ", eval0_8["metric_mean"])
    # print()
    
    # print("0.2 metric_results: ", eval0_2["metric_results"][:5])
    # print("0.5 metric_results: ", eval0_5["metric_results"][:5])
    # print("0.8 metric_results: ", eval0_8["metric_results"][:5])
    
    fail_to_correct = []
    correct_to_fail = []
    for idx, (eval0_2_item, eval0_5_item) in enumerate(zip(eval0_2["metric_results"], eval0_5["metric_results"])):
        if eval0_2_item == 0 and eval0_5_item == 1:
            # avoid retrieve to improve performance
            correct_to_fail.append(idx)
        elif eval0_2_item == 1 and eval0_5_item == 0:
            # avoid retrieve to disminish performance
            fail_to_correct.append(idx)
    
    print("The id that fail due to the tolerant of retrieved threshold: ", correct_to_fail)
    print("len of correct_to_fail: ", len(correct_to_fail), "\n")
    print("The id that performs improved due to more chance to do retrieval: ", fail_to_correct)
    print("len of fail_to_correct: ", len(fail_to_correct))
    print("len of total: ", len(eval0_2["metric_results"]))
    
    
    # Process eval data
    # eval_data = preprocess_input_data(eval, task=None)
    example_id = 5
    print("Check out the detail of correct_to_fail through example {id}: ".format(id=example_id))
    
    print("eval0_2 preds: ", eval0_2["preds"][example_id])
    print("eval0_5 preds: ", eval0_5["preds"][example_id])
    print("eval0_2 prompts: ", eval0_2["prompts"][example_id])
    # print("eval0_5 prompts: ", eval0_5["prompts"][example_id])
    print("eval0_2 do_retrieve_judge: ", eval0_2["do_retrieve_judge"][example_id])
    print("eval0_5 do_retrieve_judge: ", eval0_5["do_retrieve_judge"][example_id])
    print("eval0_2_all_results: ", eval0_2["all_results"][example_id])
    # print("eval0_5_all_results: ", eval0_5["all_results"][example_id])
    print("eval0_2 evidence_augmented_inputs: ", eval0_2["evidence_augmented_inputs"][example_id])
    # print("eval0_5 evidence_augmented_inputs: ", eval0_5["evidence_augmented_inputs"][example_id])
    print("Ex31 golds: ", eval0_2["golds"][example_id])
    

if __name__ == "__main__":
    main()
