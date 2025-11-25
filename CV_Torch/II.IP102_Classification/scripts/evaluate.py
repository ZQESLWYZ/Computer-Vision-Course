import torch
import numpy
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
#from sklearn.exceptions import UndefinedMetricWarning

#warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def evaluate(epoch, model, dataloader, lossfunction, device, mode, eval_batch, writer=None):
    
    model.eval()
    running_loss = 0.0
    all_pred = []
    all_target = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i > int(eval_batch):
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = lossfunction(outputs, targets)
            
            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            
            all_target.extend(targets.cpu().numpy())
            all_pred.extend((predicted.cpu().numpy()))
            
        total_loss = running_loss / len(5)
        accuracy = accuracy_score(all_target, all_pred)
        f1 = f1_score(all_target, all_pred, average='weighted')
        precision = precision_score(all_target, all_pred, average='weighted')
        recall = recall_score(all_target, all_pred, average='weighted')
    
    if writer:
        writer.add_scalar(f'{mode}/loss', total_loss, epoch)
        writer.add_scalar(f'{mode}/accuracy', accuracy, epoch)
        writer.add_scalar(f'{mode}/f1', f1, epoch)
        writer.add_scalar(f'{mode}/recall', recall, epoch)
        writer.add_scalar(f'{mode}/precision', precision, epoch)
        
    
    print(f"Epoch: {epoch}")
    print(f"{mode} metrics: Accuracy: {accuracy}, F1-Score: {f1}, Precision: {precision}, Recall: {recall}, Loss: {total_loss}")