LORA_DIR := ../comfy/ComfyUI/models/loras

.PHONY: lora sync

lora:
	accelerate launch --num_cpu_threads_per_process 3 --mixed_precision bf16 anima_train_network.py \
		--config_file training_config.toml

sync:
	cp output/*.safetensors $(LORA_DIR)/
