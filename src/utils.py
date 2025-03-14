import torch 
from tqdm import tqdm
import Levenshtein

def get_batch_metrics(tokenizer, pred_ids, decoder_target_ids):

    pred_texts = [tokenizer.decode(pred_ids[j].cpu().tolist(), skip_special_tokens=True) for j in  range(pred_ids.shape[0])]
    
    gt_texts = [tokenizer.decode(decoder_target_ids[j].cpu().tolist(), skip_special_tokens=True) for j in range(pred_ids.shape[0])]

    correct = sum([1 for pred_text, gt_text in zip(pred_texts, gt_texts) if pred_text == gt_text])
    total = len(pred_texts)
    accuracy = correct / total * 100
    
    dists = [calculate_cer(pred_text, gt_text) for pred_text, gt_text in zip(pred_texts, gt_texts)]
    cer = sum(dists) / total

    return {'acc':accuracy, 'cer':cer}

def calculate_cer(pred_text, gt_text):
    return Levenshtein.distance(pred_text, gt_text) / len(gt_text) if len(gt_text) > 0 else 0

def beam_search_evaluate(epoch, model, test_loader, tokenizer, device, beam_size=5):
    """Evaluate the model using beam search decoding."""
    model.eval()
    epoch_test_metrics = {
        'batch_accs': [],
        'batch_cers': [],
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Epoch {epoch} [Beam Search Evaluation]")):
            images, input_ids, _ = batch
            images = images.to(device)
            pred_ids = model.generate(images, 
                                      start_token_id=tokenizer.convert_tokens_to_ids(tokenizer.bos_token), 
                                      end_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
                                      pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
                                      max_length=128,
                                      beam_size=beam_size)
            decoder_target_ids = input_ids[:, 1:].to(device)
            batch_metrics = get_batch_metrics(tokenizer, pred_ids, decoder_target_ids)
            epoch_test_metrics['batch_accs'].append(batch_metrics['acc'])
            epoch_test_metrics['batch_cers'].append(batch_metrics['cer'])
    
    avg_epoch_test_acc = sum(epoch_test_metrics['batch_accs']) / len(epoch_test_metrics['batch_accs'])
    avg_epoch_test_cer = sum(epoch_test_metrics['batch_cers']) / len(epoch_test_metrics['batch_cers'])
    
    samples = {'pred_ids':pred_ids, 'target_ids':decoder_target_ids}
    
    return avg_epoch_test_acc, avg_epoch_test_cer, samples