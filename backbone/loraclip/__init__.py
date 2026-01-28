from .lora_clip import *


def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	trainable_params_names = []

	for name, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
			trainable_params_names.append(name)
			# if "lora_B" in name:
			# 	print(param.sum())
	
	print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

	print("Trainable Parameters:")
	for name in trainable_params_names:
		print(name)
	# check = True
	# for name in trainable_params_names:
	# 	if "lora" not in name:
	# 		check = False
	
	# if check:
	# 	print("Are LoRA parameters correctly present?  Yes")
	# else:
	# 	print("Are LORA parameters correctly present?  No")