python run_classifier.py --data_dir ./sentiment_data/ --bert_config_file ../chinese_L-12_H-768_A-12/bert_config.json --task_name senti --vocab_file ../chinese_L-12_H-768_A-12/vocab.txt --output_dir ./result/ --init_checkpoint ../chinese_L-12_H-768_A-12/bert_model.ckpt --do_train True --do_eval True --do_predict True --save_checkpoints_steps 500 --iterations_per_loop 500
