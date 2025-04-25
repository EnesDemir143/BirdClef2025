import torch


def train_model(model, data_loader, validation_loader, device, loss_fn, optimiser, EPOCHS=30):
    print(f"Starting training for {EPOCHS} epochs")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_val_loss = 0
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 15)
        
        for inputs, labels in data_loader:
            batch_count += 1
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            total_loss += loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            if batch_count % 10 == 0:
                print(f"Batch {batch_count}: Loss = {loss.item():.4f}")
            
            model.eval()
            with torch.no_grad():
                val_batch_count = 0
                for val_input, val_label in validation_loader:
                    val_batch_count += 1
                    val_input, val_label = val_input.to(device), val_label.to(device)
                    val_predictions = model(val_input)
                    val_loss = loss_fn(val_predictions, val_label)
                    total_val_loss += val_loss
        
        avg_train_loss = total_loss / batch_count
        avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
        print(f"Epoch {epoch+1} completed - Avg Train Loss: {avg_train_loss.item():.4f}, Avg Validation Loss: {avg_val_loss.item():.4f}")
    
    print("\nTraining completed!")