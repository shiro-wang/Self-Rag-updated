python run_short_form.py \
--model_name selfrag/selfrag_llama2_7b \
--input_file ../../../../dataspace/P76124574/SELF-RAG/eval_data/popqa_longtail_w_gs.jsonl \
--mode adaptive_retrieval \
--max_new_tokens 100 \
--threshold 0.2 \
--output_file ../../../../dataspace/P76124574/SELF-RAG/eval_output/popqa_longtail_w_gs_retprob_force100.json \
--metric match --ndocs 10 --use_groundness --use_utility --use_seqscore \
--dtype half