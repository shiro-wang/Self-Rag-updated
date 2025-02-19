import json
import os
import numpy as np
from utils import load_jsonlines
from run_short_form import preprocess_input_data

def main():
    cmp1_path = "../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_retprob_forceall_last.json"
    cmp2_path = "../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_retprob_all.json"
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
    
    cmp1_correct_cmp2_fail = []
    cmp1_fail_cmp2_correct = []
    for idx, (cmp1_item, cmp2_item) in enumerate(zip(cmp1["metric_results"], cmp2["metric_results"])):
        if cmp1_item == 0 and cmp2_item == 1:
            # avoid retrieve to improve performance
            cmp1_fail_cmp2_correct.append(idx)
        elif cmp1_item == 1 and cmp2_item == 0:
            # avoid retrieve to disminish performance
            cmp1_correct_cmp2_fail.append(idx)
    
    print("The length of data: ", len(cmp2["metric_results"]))
    print("The id that failed in cmp1 but correct in cmp2: ", cmp1_fail_cmp2_correct)
    print("len of cmp1_fail_cmp2_correct: ", len(cmp1_fail_cmp2_correct), "\n")
    print("The id that failed in cmp2 but correct in cmp1: ", cmp1_correct_cmp2_fail)
    print("len of cmp1_correct_cmp2_fail: ", len(cmp1_correct_cmp2_fail))
    
    # print("The detail of cmp1_fail_cmp2_correct: ")
    # for idx in cmp1_fail_cmp2_correct:
    #     print("cmp1 preds: ", cmp1["preds"][idx])
    #     print("cmp2 preds: ", cmp2["preds"][idx])
    #     print("cmp1 do_retrieve_judge: ", cmp1["do_retrieve_judge"][idx])
    #     print("cmp2 do_retrieve_judge: ", cmp2["do_retrieve_judge"][idx])
    #     print("cmp1 retrieve_probs: ", cmp1["retrieve_probs"][idx])
    #     print("cmp2 retrieve_probs: ", cmp2["retrieve_probs"][idx])
    #     print("Ex{0} golds: {1}".format(idx, cmp1["golds"][idx]))
    #     print()
        
    # print("The detail of cmp1_correct_cmp2_fail: ")
    # for idx in cmp1_correct_cmp2_fail:
    #     print("cmp1 preds: ", cmp1["preds"][idx])
    #     print("cmp2 preds: ", cmp2["preds"][idx])
    #     print("cmp1 do_retrieve_judge: ", cmp1["do_retrieve_judge"][idx])
    #     print("cmp2 do_retrieve_judge: ", cmp2["do_retrieve_judge"][idx])
    #     print("cmp1 retrieve_probs: ", cmp1["retrieve_probs"][idx])
    #     print("cmp2 retrieve_probs: ", cmp2["retrieve_probs"][idx])
    #     print("Ex{0} golds: {1}".format(idx, cmp1["golds"][idx]))
    #     print()
    
    
    # Process eval data
    # eval_data = preprocess_input_data(eval, task=None)
    example_id = 9
    print("Check out the detail of cmp1_fail_cmp2_correct through example {id}: ".format(id=example_id))
    
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
    print("Ex{0} golds: {1}".format(example_id, cmp1["golds"][example_id]))
    
    # do_retrieve_judge_diff
    do_retrieve_judge_diff = []
    for question in range(len(cmp1["retrieve_probs"])):
        if cmp1["do_retrieve_judge"][question] != cmp2["do_retrieve_judge"][question]:
            do_retrieve_judge_diff.append((question, cmp1["do_retrieve_judge"][question], cmp2["do_retrieve_judge"][question]))
    print("\nThe length of retrieve_probs_diff: ", len(do_retrieve_judge_diff))
    
    # retrieve_probs_diff
    retrieve_probs_diff = []
    retrieve_probs_diff_value = 0
    for question in range(len(cmp1["retrieve_probs"])):
        retrieve_probs_diff_value += cmp1["retrieve_probs"][question] - cmp2["retrieve_probs"][question]
        if cmp1["retrieve_probs"][question] > cmp2["retrieve_probs"][question]:
            retrieve_probs_diff.append((question, cmp1["retrieve_probs"][question], cmp2["retrieve_probs"][question]))
    print("The length of retrieve_probs_diff: ", len(retrieve_probs_diff))
    print("The mean of retrieve_probs_diff_value: ", retrieve_probs_diff_value / len(cmp1["retrieve_probs"]))
    # print("retrieve_probs_diff: ", retrieve_probs_diff)
    
    print("Metrics result of cmp1:", np.mean(cmp1["metric_results"]))
    print("Metrics result of cmp2:", np.mean(cmp2["metric_results"]))
    

if __name__ == "__main__":
    main()
