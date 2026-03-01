LORA_DIR := ../comfy/ComfyUI/models/loras

.PHONY: lora sync lora-mini merge

lora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 anima_train_network.py \
		--config_file training_config.toml

sync:
	cp output/*.safetensors $(LORA_DIR)/

TRAIN_DIR ?= train_datasets

lora-mini:
	python tools/train_mini_loras.py \
		--config training_mini_config.toml \
		--train_dir $(TRAIN_DIR) \
		--output_dir output_mini \
		$(if $(COUNT),--group_size $(COUNT)) \
		$(if $(filter-out $@,$(MAKECMDGOALS)),--group_size $(filter-out $@,$(MAKECMDGOALS)))

merge:
	python networks/dare_ties_merge_lora.py \
		--models $(wildcard output_mini/*.safetensors) \
		--ratios $(foreach f,$(wildcard output_mini/*.safetensors),1.0) \
		--method ties --density 0.5 --device cuda \
		--save_to output/merged_lora.safetensors

# Catch numeric args for lora-mini
%:
	@:
