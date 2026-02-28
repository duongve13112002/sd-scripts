LORA_DIR := ../comfy/ComfyUI/models/loras

.PHONY: lora sync lora-mini merge

lora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 anima_train_network.py \
		--config_file training_config.toml

sync:
	cp output/*.safetensors $(LORA_DIR)/

lora-mini:
	python tools/train_mini_loras.py \
		--config training_mini_config.toml \
		--train_dir train_datasets \
		--output_dir output_mini \
		$(if $(filter-out $@,$(MAKECMDGOALS)),--count $(filter-out $@,$(MAKECMDGOALS)))

merge:
	python networks/dare_ties_merge_lora.py \
		--models $(wildcard output_mini/*.safetensors) \
		--ratios $(foreach f,$(wildcard output_mini/*.safetensors),1.0) \
		--density 0.5 --seed 42 --device cuda --num_shards 5 \
		--save_to output/merged_lora.safetensors

# Catch numeric args for lora-mini
%:
	@:
