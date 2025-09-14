# Fine-tuning Llama-3.2 (3B Instruct)

This project demonstrates how to fine-tune the **Llama 3.2 - 3B Instruct** model using the [🤗 Transformers](https://huggingface.co/docs/transformers) library and the **QLoRA** approach on a customer support dataset.

---

## 🔑 What is QLoRA?

**QLoRA = Quantized Low-Rank Adapter**  
It enables efficient fine-tuning of large language models on consumer-grade GPUs.

QLoRA combines three key ideas:

- **4-bit quantization** → compresses model weights so the base model fits into GPU memory.  
- **LoRA adapters** → small low-rank trainable matrices are added on top of frozen quantized weights.  
- **PEFT (Parameter-Efficient Fine-Tuning)** → only the LoRA adapters are trained (a tiny % of the parameters), while the giant quantized base model stays frozen.

---

## ⚙️ Training Dynamics

- **Base weights**: quantized, frozen, and never updated.  
- **Adapters**: small FP16/BF16 matrices that are actually trained.  
- **Inference**: LoRA adapters are merged with the frozen base model, giving you an adapted model without retraining the huge parameter set.

---

## 🚀 Benefits of QLoRA

- **Memory-efficient**: fine-tune models like 7B on a 12GB GPU, or even 65B on a single 48GB GPU.  
- **Cost-effective**: only a few million parameters are trained, saving storage and compute.  
- **High quality**: achieves results competitive with full fine-tuning in many benchmarks.  

---

## 🛠️ Workflow

1. **Load the base model** in 4-bit quantized mode with `BitsAndBytesConfig`.  
2. **Prepare model for k-bit training** with `prepare_model_for_kbit_training`.  
3. **Attach LoRA adapters** via PEFT (`get_peft_model`).  
4. **Fine-tune** only the LoRA layers on your dataset.  
5. **Save adapters** (small, lightweight).  
6. **Optionally merge adapters** into the base model for deployment.  

---

## 📊 Example Use Case

We fine-tune **Llama 3.2 3B Instruct** to act as a **customer support agent** that:  
- Answers questions politely.  
- Provides consistent and helpful responses.  
- Adapts the base Llama model to domain-specific data without retraining all parameters.  

