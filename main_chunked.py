import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.tinyBERT_chunked import TinyBERTChunked as TinyBERT
from data_process_chunk import (
    get_loaders,
    LABEL2ID,
    ID2LABEL,
)


# ---- 1. Configuration ----

EPOCHS = 15
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MIN_LR = 1e-5

BEST_MODEL_PATH = "checkpoints/tinyBERT_chunk_best.pth"

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ---- 2. Device ----

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---- 3. Data ----

train_loader, valid_loader, test_loader, train_eval_loader = (
    get_loaders(
        batch_size=16,
        eval_batch_size=32,
        seed=42,
        max_length=512,
    )
)


# ---- 4. Model ----

model = TinyBERT(
    num_classes=len(LABEL2ID)
).to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ---- 5. Loss & optimizer ----

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Tested for frozen BERT
# optimizer = optim.AdamW(
#     (p for p in model.parameters() if p.requires_grad),
#     lr=1e-3,
#     weight_decay=0.01,
# )

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=MIN_LR,
)


# ---- 6. Best model tracking ----

os.makedirs(
    os.path.dirname(BEST_MODEL_PATH),
    exist_ok=True
)

best_valid_loss = float("inf")
best_epoch = 0


# ---- 7. Training loop ----

start_time = time.perf_counter()

for epoch in range(EPOCHS):

    epoch_start = time.perf_counter()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        chunk_to_article = batch["chunk_to_article"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            chunk_to_article=chunk_to_article,
            num_articles=labels.size(0),
        )

        loss = criterion(logits, labels)

        loss.backward()

        # Prevent excessively large gradients
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # Accumulate total loss, accounting for batch size
        train_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total

    train_accuracy = (
        100 * train_correct / train_total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():

        for batch in valid_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            chunk_to_article = batch["chunk_to_article"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                chunk_to_article=chunk_to_article,
                num_articles=labels.size(0),
            )

            loss = criterion(logits, labels)

            valid_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)

            valid_correct += (
                preds == labels
            ).sum().item()

            valid_total += labels.size(0)

    avg_valid_loss = valid_loss / valid_total

    valid_accuracy = (
        100 * valid_correct / valid_total
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if avg_valid_loss < best_valid_loss:

        best_valid_loss = avg_valid_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH,
        )

        print(
            f"  -> New best model saved "
            f"(Valid Loss: {best_valid_loss:.4f})"
        )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    scheduler.step()

    epoch_time = time.perf_counter() - epoch_start


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Valid Loss: {avg_valid_loss:.4f} | "
        f"Valid Acc: {valid_accuracy:.2f}% | "
        f"LR: {current_lr:.6f} | "
        f"Time: {epoch_time:.2f}s"
    )


total_time = time.perf_counter() - start_time

print(f"\nTotal training time: {total_time:.2f}s")

print(
    f"Best model: Epoch {best_epoch} "
    f"| Valid Loss: {best_valid_loss:.4f}"
)


# ---- 8. Load best model ----

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

print(f"\nLoaded best model from epoch {best_epoch}")


# ---- 9. Final Training Evaluation ----

model.eval()

train_eval_correct = 0
train_eval_total = 0

with torch.no_grad():

    for batch in train_eval_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        chunk_to_article = batch["chunk_to_article"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            chunk_to_article=chunk_to_article,
            num_articles=labels.size(0),
        )

        preds = logits.argmax(dim=1)

        train_eval_correct += (
            preds == labels
        ).sum().item()

        train_eval_total += labels.size(0)

train_eval_accuracy = (
    100 * train_eval_correct / train_eval_total
)

print(
    f"Training accuracy: "
    f"{train_eval_accuracy:.2f}%"
)


# ---- 10. Final Test Evaluation ----

test_correct = 0
test_total = 0
test_loss = 0.0

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        chunk_to_article = batch["chunk_to_article"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            chunk_to_article=chunk_to_article,
            num_articles=labels.size(0),
        )
        loss = criterion(logits, labels)

        test_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)

        test_correct += (
            preds == labels
        ).sum().item()

        test_total += labels.size(0)

test_accuracy = 100 * test_correct / test_total
avg_test_loss = test_loss / test_total

print(f"\nTest Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")