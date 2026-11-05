"""
# Script Overview
This script contains the final reusable model definitions experimented on the notebooks

## Updates
- 01_mlp_model_development.ipynb did not yield good results,
thus will not be in this script.
- CNNs will be experimented on. If successful, may update.

## Notes

Native PyTorch training loop

1. Set model to training mode
model.train()  
2. Initialise var to accumulate training loss
train_loss = 0.0
3. Loop through training data
for images, labels in train_loader:
    4. Clear gradients
    optimizer.zero_grad()
    5. Forward pass: store model's logits(predictions) for current batch
    outputs = model(images)
    6. Calculate loss
    loss = criterion(outputs, labels)
    7. Backpropagation
    loss.backward()  
    8. Use gradients to update model's weights
    optimizer.step() 
    9. Accumulate training loss
    train_loss += loss.item()
10. Calculate average epoch loss
epoch_loss = train_loss / len(train_loader)

Before that, we should have 
1. Prepared data
train_loader, val_loader
2. Defined model
model = ChestXrayMLP()
3. Initialise loss function (ONCE, before training)
criterion - nn.CrossEntropyLoss()
4. Initialise optimizer (ONCE, before training)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
5. Start training loop (every batch)
> model.train() is called at the beginning of each training epoch
"""