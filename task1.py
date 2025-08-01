import os
import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras import layers
from layers import TextVectorization

os.environ["KERAS_BACKEND"] = "tensorflow"

# Check for GPU availability and configure TensorFlow to use it
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is enabled for training")
else:
    print("No GPU found. Using CPU instead.")

# Mixed precision for faster GPU training
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Use FP16 for faster GPU computation
    print("Mixed precision enabled")
except:
    print("Mixed precision not available in your TensorFlow version")

text_file = "English-Spanish.txt"
vocab_size = 15000
sequence_length = 20
batch_size = 64
embed_dim = 256
latent_dim = 2048
num_heads = 8
epochs = 10


# Load and preprocess data
def load_text_pairs(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split("\n")
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.read().strip().split("\n")

    text_pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            eng = parts[0]
            spa = f"[start] {parts[1]} [end]"
            text_pairs.append((eng, spa))
    return text_pairs


# Train/val/test split
def split_data(pairs, val_fraction=0.15):
    random.shuffle(pairs)
    num_val_samples = int(val_fraction * len(pairs))
    num_train_samples = len(pairs) - 2 * num_val_samples
    return (
        pairs[:num_train_samples],
        pairs[num_train_samples:num_train_samples + num_val_samples],
        pairs[num_train_samples + num_val_samples:]
    )


text_pairs = load_text_pairs(text_file)
train_pairs, val_pairs, test_pairs = split_data(text_pairs)

print(f"Number of training pairs: {len(train_pairs)}")
print(f"Number of validation pairs: {len(val_pairs)}")
print(f"Number of test pairs: {len(test_pairs)}")

# Strip characters for custom standardization
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[^a-z0-9/]", "")


eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize=custom_standardization,
)

spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=None,
)

train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)


def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)  # Use parallel processing
    return dataset.cache().shuffle(2048).prefetch(tf.data.AUTOTUNE)  # Better prefetching for GPU


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

test_ds = make_dataset(test_pairs)


# -----------------------------
# Transformer Components
# -----------------------------

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Self-attention layer
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        # Cross-attention layer to attend to encoder outputs
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        # Feed-forward network
        self.dense_proj = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        # Layer normalization layers
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        # Create causal mask for decoder self-attention
        causal_mask = self.get_causal_attention_mask(inputs)

        # Handle padding mask if provided
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            # Combine padding mask with causal mask
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = causal_mask

        self_attention_output = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=padding_mask
        )
        self_attention_output = self.layernorm_1(inputs + self_attention_output)

        cross_attention_output = self.cross_attention(
            query=self_attention_output,
            value=encoder_outputs,
            key=encoder_outputs,
        )
        cross_attention_output = self.layernorm_2(
            self_attention_output + cross_attention_output
        )

        proj_output = self.dense_proj(cross_attention_output)
        return self.layernorm_3(cross_attention_output + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)


# -----------------------------
# Build Model
# -----------------------------
encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = tf.keras.Input(shape=(None, embed_dim), name="encoder_outputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder(
    [decoder_inputs, encoder_outputs],
)
transformer = tf.keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

# -----------------------------
# Train Model
# -----------------------------

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

transformer.summary()
transformer.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath="transformer_checkpoint.h5",
        save_best_only=True,
        monitor="val_loss"
    ),
    tf.keras.callbacks.TensorBoard(log_dir="./logs")
]

history = transformer.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

test_loss, test_accuracy = transformer.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# -----------------------------
# Inference
# -----------------------------

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"

    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        # Get the predicted token index
        sampled_token_index = tf.argmax(predictions[0, i, :]).numpy()
        sampled_token = spa_index_lookup[sampled_token_index]

        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break

    clean_sentence = decoded_sentence.replace("[start] ", "").replace(" [end]", "")
    return clean_sentence


English_sentence1 = "Deep Learning is widely used in Natural Language Processing, as Dr. Sun said in CSC 446/646."
English_sentence2 = "Natural language is how humans speak and write."

translated_sentence1 = decode_sequence(English_sentence1)
print("Sentence 1 (English):", English_sentence1)
print("Translation 1 (Spanish):", translated_sentence1)

translated_sentence2 = decode_sequence(English_sentence2)
print("Sentence 2 (English):", English_sentence2)
print("Translation 2 (Spanish):", translated_sentence2)

transformer.save('transformer_translator_model')


def plot_training_history(history):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


    plot_training_history(history)

# Optional: For attention visualization (bonus question)
class TranslationAttention(tf.keras.Model):
    def __init__(self, transformer, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer
        # Extract the encoder and decoder from the transformer
        self.encoder = transformer.get_layer('model')
        self.decoder = transformer.get_layer('model_1')

    def call(self, inputs):
        # Run the forward pass and extract attention weights
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder([decoder_inputs, encoder_outputs])

        # Get cross-attention weights (this requires additional code to extract from the model)
        # This is a placeholder - actual implementation depends on TF version and model structure

        return {
            'outputs': decoder_outputs,
            'encoder_outputs': encoder_outputs
        }


transformer_with_attention = TranslationAttention(transformer)