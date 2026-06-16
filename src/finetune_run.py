import keras
import keras_hub
import os

os.environ["KERAS_BACKEND"] = "jax"

print("Lade Gemma 4 E2B für LoRA Fine-Tuning...")
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

# 2. Aktivierung von Low-Rank Adaptation (LoRA)
gemma_lm.backbone.enable_lora(rank=4)

# 3. Dummy-Trainingsdaten formatieren
training_data = [
    {"prompt": "Status: queue_length=14. Action?", "response": "NEW_SERVICE_RATE: 3.0"},
    {"prompt": "Status: queue_length=1. Action?", "response": "NEW_SERVICE_RATE: 0.5"}
]
prompts = [f"{item['prompt']} -> {item['response']}" for item in training_data]

# 4. Kompilierung des Modells mit optimierter Lernrate
gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=2e-4),
    weighted_metrics=["accuracy"]
)

# 5. Training für eine repräsentative Epoche
gemma_lm.fit(prompts, epochs=1, batch_size=2)

# 6. Sichern des trainierten Adapters im VM-Speicher
gemma_lm.save_to_preset("./gemma4_lora_adapter")
print("Feintuning erfolgreich abgeschlossen. Gewichte lokal in der Colab-VM gespeichert.")