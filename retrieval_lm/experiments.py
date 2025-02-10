import json
import os
from utils import load_jsonlines
from run_short_form import preprocess_input_data

def main():
    cmp1_path = "../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_retprob_force100.json"
    cmp2_path = "../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_retprob.json"
    # eval0_8_path = "../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_trs0-8.json"
    # eval_path = "../../../../dataspace/P76124574/SELF-RAG/eval_data/popqa_longtail_w_gs.jsonl"
    
    cmp1 = json.load(open(cmp1_path))
    cmp2 = json.load(open(cmp2_path))
    # eval0_8 = json.load(open(eval0_8_path))
    # eval = load_jsonlines(eval_path)
    
    # print("cmp1 metric_mean: ", cmp1["metric_mean"])
    # print("cmp2 metric_mean: ", cmp2["metric_mean"])
    # print("0.8 metric_mean: ", eval0_8["metric_mean"])
    # print()
    
    # print("cmp1 metric_results: ", cmp1["metric_results"][:5])
    # print("cmp2 metric_results: ", cmp2["metric_results"][:5])
    # print("0.8 metric_results: ", eval0_8["metric_results"][:5])
    
    fail_to_correct = []
    correct_to_fail = []
    for idx, (cmp1_item, cmp2_item) in enumerate(zip(cmp1["metric_results"], cmp2["metric_results"])):
        if cmp1_item == 0 and cmp2_item == 1:
            # avoid retrieve to improve performance
            correct_to_fail.append(idx)
        elif cmp1_item == 1 and cmp2_item == 0:
            # avoid retrieve to disminish performance
            fail_to_correct.append(idx)
    
    print("The length of data: ", len(cmp2["metric_results"]))
    print("The id that failed in cmp1 but correct in cmp2: ", correct_to_fail)
    print("len of correct_to_fail: ", len(correct_to_fail), "\n")
    print("The id that failed in cmp2 but correct in cmp1: ", fail_to_correct)
    print("len of fail_to_correct: ", len(fail_to_correct))
    print("len of total: ", len(cmp1["metric_results"]))
    
    
    # Process eval data
    # eval_data = preprocess_input_data(eval, task=None)
    example_id = 8
    print("Check out the detail of correct_to_fail through example {id}: ".format(id=example_id))
    
    print("cmp1 preds: ", cmp1["preds"][example_id])
    print("cmp2 preds: ", cmp2["preds"][example_id])
    print("cmp1 prompts: ", cmp1["prompts"][example_id])
    print("cmp2 prompts: ", cmp2["prompts"][example_id])
    print("cmp1 do_retrieve_judge: ", cmp1["do_retrieve_judge"][example_id])
    print("cmp2 do_retrieve_judge: ", cmp2["do_retrieve_judge"][example_id])
    print("cmp1 retrieve_probs: ", cmp1["retrieve_probs"][example_id])
    print("cmp2 retrieve_probs: ", cmp2["retrieve_probs"][example_id])

    # print("cmp1_all_results: ", cmp1["all_results"][example_id])
    # print("cmp2_all_results: ", cmp2["all_results"][example_id])
    # print("cmp1 evidence_augmented_inputs: ", cmp1["evidence_augmented_inputs"][example_id])
    # print("cmp2 evidence_augmented_inputs: ", cmp2["evidence_augmented_inputs"][example_id])
    print("Ex31 golds: ", cmp1["golds"][example_id])
    
    retrieve_probs_diff = []
    for question in range(len(cmp1["retrieve_probs"])):
        if cmp1["retrieve_probs"][question] != cmp2["retrieve_probs"][question]:
            retrieve_probs_diff.append((question, cmp1["retrieve_probs"][question], cmp2["retrieve_probs"][question]))
    print("The length of retrieve_probs_diff: ", len(retrieve_probs_diff))
    print("retrieve_probs_diff: ", retrieve_probs_diff)
    

if __name__ == "__main__":
    main()
