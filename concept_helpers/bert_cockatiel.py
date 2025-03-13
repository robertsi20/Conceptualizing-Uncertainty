from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from tqdm import tqdm

import torch
from sklearn.metrics import precision_recall_fscore_support

class BertForCockatiel(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.activation = torch.nn.ReLU()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        pooled_output = outputs[1]
        pooled_output = self.activation(pooled_output)
        output_dropout = self.dropout(pooled_output)
        logits = self.classifier(output_dropout)

        outputs = (logits,) + (pooled_output,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs[0],) + (pooled_output,) + outputs[2:]

        return outputs  # (loss), logits, pooled output, ...


class DatasetForTransformer(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class BertCockatielWrapper():

    def __init__(self, num_labels: int, model_name: str = None, batch_size: int = 16, verbose: bool = False,
                 pooling: str = 'mean', optimizer: torch.optim.Optimizer = None,
                 loss_function: torch.nn.modules.loss._Loss = None, lr=1e-5, class_weights=None):
        self.model = None
        self.tokenizer = None
        self.num_labels = num_labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.verbose = verbose
        self.prepare(model_name=model_name, batch_size=batch_size, verbose=verbose)

        self.lr = lr
        if optimizer is not None:
            print("use custom optimizer")
            print(optimizer)
            self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        else:
            print("no optmizer specified, default to AdamW")
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        if loss_function is None:
            print("no loss function specified, default to cross entropy")
            loss_function = torch.nn.CrossEntropyLoss

        if class_weights is not None:
            print("use positive class weights")
            class_weights = torch.tensor(class_weights)
            if torch.cuda.is_available():
                class_weights = class_weights.to('cuda')
            self.loss_function = loss_function(pos_weight=class_weights)
        else:
            self.loss_function = loss_function()

    def __switch_to_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            print('Using Bert with CUDA/GPU')
        else:
            print('WARNING! Using Bert on CPU!')

    def prepare(self, **kwargs):
        model_name = kwargs.pop('model_name') or 'bert-base-uncased'

        self.model = BertForCockatiel.from_pretrained(model_name, return_dict=True,
                                                                        num_labels=self.num_labels,
                                                                        output_hidden_states=True,
                                                                        output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, max_length=self.model.config.max_position_embeddings,
                                                       truncation=True)
        if self.tokenizer.pad_token is None:
            print("manually define padding token for model %s" % model_name)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.tokenizer.padding_side = "left"  # for generator models
        self.__switch_to_cuda()
        self.model.eval()

    def save(self, path: str):
        self.model.save_pretrained(path)

    def load(self, path: str):
        self.model = self.model.to('cpu')
        print('Loading existing model...')
        self.model = BertForCockatiel.from_pretrained(path)
        self.__switch_to_cuda()
        self.model.eval()

    def embed(self, texts: list[str], verbose=False):
        max_length = 512
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True, 
                                padding='max_length')
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        outputs = []

        dbg_count = 0
        for batch in tqdm(loader, leave=True):
            if torch.cuda.is_available():
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda')

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            out = self.model.forward(input_ids, attention_mask=attention_mask)
            #pooled_emb = self.model.activation(out.pooler_output)
            pooled_emb = out[1]
            
            outputs.append(pooled_emb.to('cpu').detach().numpy())

            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')

            del input_ids
            del attention_mask
            del pooled_emb
            torch.cuda.empty_cache()

        outputs = np.vstack(outputs)
        return outputs

    def predict(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        outputs = []

        for batch in tqdm(loader, leave=True):
            if torch.cuda.is_available():
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda')

                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                out = self.model(input_ids, attention_mask=attention_mask)
                out = out[0].to('cpu')

                out = out.detach().numpy()
                outputs.append(out)

                input_ids = input_ids.to('cpu')
                attention_mask = attention_mask.to('cpu')

                del input_ids
                del attention_mask
                torch.cuda.empty_cache()
        return np.vstack(outputs)

    def retrain(self, texts: list[str], labels, epochs: int = 2):
        inputs = self.tokenizer(texts, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')
        inputs['labels'] = torch.tensor(labels)
        dataset = DatasetForTransformer(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        losses = []
        for ep in range(epochs):
            losses.append(self.retrain_one_epoch(loader, epoch=ep))
        return losses

    def retrain_one_epoch(self, loader: torch.utils.data.DataLoader, epoch: int = 0):
        overall_loss = 0
        self.model.train()

        # pull all tensor batches required for training
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = self.model.forward(input_ids, attention_mask=attention_mask)
            # extract loss
            loss = self.loss_function(outputs[0], labels)

            # calculate loss for every parameter that needs grad update
            loss.backward()

            # update parameters
            self.optimizer.step()

            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

            loss = loss.detach().item()
            overall_loss += loss

            outputs[0].to('cpu')
            input_ids = input_ids.to('cpu')
            attention_mask = attention_mask.to('cpu')
            labels = labels.to('cpu')
            del input_ids
            del attention_mask
            del labels

        self.model.eval()
        torch.cuda.empty_cache()
        return overall_loss


    def features(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
        ):
        return self.model.forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states)[1]