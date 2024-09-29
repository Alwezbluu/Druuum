from datasets import load_dataset, Audio
import numpy as np
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate

gtzan = load_dataset("marsyas/gtzan","all", trust_remote_code=True)
gtzan = gtzan["train"].train_test_split(seed=42,shuffle=True, test_size=0.1)

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True, return_attention_mask=True, trust_remote_code=True)
sampling_rate = feature_extractor.sampling_rate
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
sample = gtzan["train"][0]["audio"]
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
max_duration = 20.0


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs


gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=25,
    num_proc=1,
)

gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
id2label_fn = gtzan["train"].features["genre"].int2str
id2label = {
	str(i): id2label_fn(i)
	for i in range(len(gtzan_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

model = AutoModelForAudioClassification.from_pretrained(
	model_id,
	num_labels=num_labels,
	label2id=label2id,
	id2label=id2label,
	trust_remote_code=True
)

model_name = model_id.split("/")[-1]
batch_size = 2
gradient_accumulation_steps = 1
num_train_epochs = 5

training_args = TrainingArguments(
	f"{model_name}-Music classification Finetuned",
	evaluation_strategy="epoch",
	save_strategy="epoch",
	learning_rate=5e-5,
	per_device_train_batch_size=batch_size,
	gradient_accumulation_steps=gradient_accumulation_steps,
	per_device_eval_batch_size=batch_size,
	num_train_epochs=num_train_epochs,
	warmup_ratio=0.1,
	logging_steps=5,
	load_best_model_at_end=True,
	metric_for_best_model="accuracy",
	fp16=True,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("/model/Saved_Model")
feature_extractor.save_pretrained("/model/feature_extractor_savedModel")
