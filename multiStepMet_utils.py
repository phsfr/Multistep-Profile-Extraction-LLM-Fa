import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

import torch.nn as nn
from tqdm import tqdm
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel, XLMRobertaTokenizer


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointXLMRoberta(XLMRobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointXLMRoberta, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = XLMRobertaModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=None)

        pooled_output = outputs[1]  # [CLS]
        sequence_output = outputs[0]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]

        outputs = (total_loss,) + outputs

        return outputs


def convert_examples_to_features(lines, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask = [], [], [], []
    for line in lines:
        tokens = []
        prev_tokens = []
        slot_labels_ids = []
        for word in line[0].split():
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))
        for prev_word in line[1].split():
            prev_word_tokens = tokenizer.tokenize(prev_word)
            if not prev_word_tokens:
                prev_word_tokens = [unk_token]
            prev_tokens.extend(prev_word_tokens)

        # Account for [CLS] and [SEP]
        special_tokens_count = 3
        input_tok_size = len(tokens) + len(prev_tokens)

        max_limit = max_seq_len - special_tokens_count
        if input_tok_size > max_limit:
            if max_limit >= len(prev_tokens):
                remain = max_limit - len(prev_tokens)
                prev_tokens = prev_tokens[-remain:]
            else:
                prev_tokens = []
                if len(tokens) > max_limit:
                    tokens = tokens[:max_limit]  # [:(max_seq_len - special_tokens_count)]
                    slot_labels_ids = slot_labels_ids[:max_limit]  # [:(max_seq_len - special_tokens_count)]

        final_tokens = [cls_token] + prev_tokens + [sep_token] + tokens + [sep_token]
        slot_labels_ids = [pad_token_label_id] + ([pad_token_label_id] * len(prev_tokens)) + [
            pad_token_label_id] + slot_labels_ids + [pad_token_label_id]
        token_type_ids = [cls_token_segment_id] + ([sequence_a_segment_id] * (len(prev_tokens) + 1)) + (
                [sequence_a_segment_id + 1] * (len(tokens) + 1))

        input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_labels_ids)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)
    return dataset


def predict_slot_intent_withPrev(input_utterances, inp_dataloader, device, model, intent_label_list,
                                 slot_label_list, pad_token_label_id):
    intent_preds = None
    slot_preds = None
    all_slot_label_mask = None

    for batch in tqdm(inp_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'intent_label_ids': None,
                      'slot_labels_ids': None,
                      'token_type_ids': batch[2]}
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

        # Intent prediction
        if intent_preds is None:
            intent_preds = intent_logits.detach().cpu().numpy()
        else:
            intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

        # Slot prediction
        if slot_preds is None:
            slot_preds = slot_logits.detach().cpu().numpy()
            all_slot_label_mask = batch[3].detach().cpu().numpy()
        else:
            slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
            all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)

    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(slot_label_list)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    final_nlu_output = []
    for pred, inp_utter, intent_pred in zip(slot_preds_list, input_utterances, intent_preds):
        intent_pred_itm = intent_label_list[intent_pred]
        tokens = inp_utter[0].split()
        assert len(tokens) == len(pred)
        output_str = '|' + intent_pred_itm + '|' + ' -> '
        for tok, prd in zip(tokens, pred):
            if prd != 'O':
                output_str += ' ' + '[' + tok + ':' + prd + ']'
            else:
                output_str += ' ' + tok
        final_nlu_output.append(output_str.strip())
    return final_nlu_output


class Args:
    def __init__(self):
        self.dropout_rate = 0.1
        self.ignore_index = 0
        self.max_seq_len = 65
        self.batch_size = 128
        self.slot_loss_coef = 1.0


def get_slot_values(nlu_output):
    subj_map = {'job': 'شغل', 'hobby':'سرگرمی', 'resistance': 'محل سکونت', 'family': 'خانواده', 'age': 'سن', 'gender': 'جنسیت',
                'name': 'اسم', 'marriage': 'وضعیت تاهل', 'education': 'تحصیلات', 'other': 'سایر', 'goodby': 'خداحافظی', 'greeting': 'احوالپرسی',
    'UNK':'نامشخص'}
    subj, sentence = nlu_output.split('->')
    subj = subj.strip()[1:-1]  # Extracting the subject
    subj = subj_map[subj]
    print(subj)
    words = sentence.strip().split()  # Splitting the sentence into words
    tag_value = ''
    last_tag = ''
    slot_vals = {}

    for word in words:
        if word.startswith('['):
            word = word[1:-1]  # Removing the surrounding brackets [ ]
            tok, tag = word.split(':')  # Splitting the token and tag

            if tag.startswith('B-'):  # Beginning of a new tag
                if len(tag_value) > 0:
                    # Add the previous tag_value to the slot_vals before starting a new one
                    if last_tag[2:] in slot_vals:
                        slot_vals[last_tag[2:]].append(tag_value.strip())
                    else:
                        slot_vals[last_tag[2:]] = [tag_value.strip()]
                # Start a new tag_value with the current token
                tag_value = tok
                last_tag = tag  # Remember the last tag

            elif tag.startswith('I-') and last_tag and tag[2:] == last_tag[2:]:  # Continuation of the previous tag
                tag_value += ' ' + tok  # Append the current token to tag_value

    # Final check to add the last accumulated tag_value
    if len(tag_value) > 0 and last_tag:
        if last_tag[2:] in slot_vals:
            slot_vals[last_tag[2:]].append(tag_value.strip())
        else:
            slot_vals[last_tag[2:]] = [tag_value.strip()]

    return slot_vals, subj


args = Args()
device = "cuda" if torch.cuda.is_available() else "cpu"
slot_file_path = 'data/slot_label.txt'
intent_file_path = 'data/intent_label.txt'

intent_label_list = [label.strip() for label in open(intent_file_path, 'r', encoding='utf-8')]
slot_label_list = [label.strip() for label in open(slot_file_path, 'r', encoding='utf-8')]
args.max_seq_len = 110
model_path = 'Path_to_Trained_SFID_Model'

model = JointXLMRoberta.from_pretrained(model_path,
                                        args=args,
                                        intent_label_lst=intent_label_list,
                                        slot_label_lst=slot_label_list, use_safetensors=True)
model.to(device)
model.eval()

pad_token_label_id = args.ignore_index
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)


def get_nlu_prediction(input_utterances):
    inp_dataset = convert_examples_to_features(input_utterances, args.max_seq_len, tokenizer,
                                               pad_token_label_id=pad_token_label_id)
    inp_dataloader = DataLoader(inp_dataset, batch_size=args.batch_size)
    nlu_output = predict_slot_intent_withPrev(input_utterances, inp_dataloader, device, model, intent_label_list,
                                              slot_label_list, pad_token_label_id)

    return nlu_output
