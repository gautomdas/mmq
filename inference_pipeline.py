import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from transformers import BertTokenizer, AutoProcessor
import numpy as np

class InferencePipeline:
    def __init__(self, model, device, processor=None):
        self.model = model
        self.processor = processor
        self.device = device

    def run_inference(self, dataset, task, **kwargs):
        if task == 'image_captioning':
            return self._run_image_captioning(dataset, kwargs.get("max_samples", None))
        elif task == 'retrieval':
            return self._run_retrieval(dataset, kwargs.get("k_test", 128))
        else:
            raise ValueError(f"Unsupported task: {task}")

    def _run_image_captioning(self, dataset, max_samples):
        results = []
        references = []
        
        for i in tqdm(range(min(len(dataset), max_samples or len(dataset)))):
            image = dataset[i][0]
            captions = dataset[i][1]
            img_id = dataset.ids[i]
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
            
            results.append({"image_id": img_id, "caption": caption})
            references.append(captions)

        return {
            'predictions': results,
            'references': references
        }

    def _run_retrieval(self, dataset, k_test):
        with torch.no_grad():
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

            texts = dataset.text
            num_text = len(texts)
            text_bs = 1
            text_ids = []
            text_embeds = []
            text_atts = []
            model = self.model
            print(self.model.device)
            print("Tokenizing captions")
            for i in tqdm(range(0, num_text, text_bs)):
                text = texts[i : min(num_text, i + text_bs)]
                text_input = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=35,
                    return_tensors="pt"
                ).to(self.model.device)

                query_embeds = model.embeddings(text_input.input_ids)
                text_output = model.qformer(
                    query_embeds=query_embeds,
                    query_length=0,
                    attention_mask=text_input.attention_mask,
                    return_dict=True
                )
                text_feat = text_output.last_hidden_state[:, 0, :]
                text_embed = F.normalize(model.text_projection(text_feat))
                text_embeds.append(text_embed)
                text_ids.append(text_input.input_ids)
                text_atts.append(text_input.attention_mask)

            text_embeds = torch.cat(text_embeds, dim=0)
            text_ids = torch.cat(text_ids, dim=0)
            text_atts = torch.cat(text_atts, dim=0)

            vit_feats = []
            image_embeds = []
            print("Getting image features")
            for samples in tqdm(dataset):
                image = samples["image"]
                image = samples["image"].unsqueeze(0).to(model.device)

                #image = processor(images=image, text=None, return_tensors="pt")["pixel_values"].to(model.device)
                vision_outputs = model.vision_model(
                    pixel_values=image,
                    return_dict=True
                )
                vit_feat = vision_outputs[0]
                image_attention_mask = torch.ones(vit_feat.size()[:-1], dtype=torch.long, device=model.device)
                query_tokens = model.query_tokens.expand(vit_feat.shape[0], -1, -1)
                query_outputs = model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=vit_feat,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True
                )
                del image_attention_mask
                image_feat = query_outputs.last_hidden_state
                image_embed = F.normalize(model.vision_projection(image_feat), dim=-1)

                vit_feats.append(vit_feat.cpu())
                image_embeds.append(image_embed)
                
                del image
                del image_feat

            vit_feats = torch.cat(vit_feats, dim=0)
            image_embeds = torch.cat(image_embeds, dim=0)

            sims_matrix = []
            for image_embed in image_embeds:
                sim_q2t = image_embed @ text_embeds.t()
                sim_i2t, _ = sim_q2t.max(0)
                sims_matrix.append(sim_i2t)
            sims_matrix = torch.stack(sims_matrix, dim=0)

            del image_embeds

            score_matrix_i2t = torch.full(
                (len(dataset.image), len(texts)), -100.0
            ).to(model.device)

            print("Calculating sim matrix")

            def compute_itm(image_inputs, text_ids, text_atts):
                image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(image_inputs.device)
                query_tokens = model.query_tokens.expand(image_inputs.shape[0], -1, -1)
                query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_inputs.device)
                attention_mask = torch.cat([query_attention_mask, text_atts], dim=1)

                del query_attention_mask, text_atts

                query_embeds = model.embeddings(
                    input_ids=text_ids,
                    query_embeds=query_tokens
                )

                query_length = query_tokens.shape[1]
                del text_ids, query_tokens

                output_itm = model.qformer(
                    query_embeds=query_embeds,
                    query_length=query_length,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_inputs,
                    encoder_attention_mask=image_atts,
                    return_dict=True
                )
                vl_embeddings = output_itm.last_hidden_state[:, : query_length, :]
                del output_itm
                itm_logit = model.itm_head(vl_embeddings)
                itm_logit = itm_logit[:, :, 1].mean(dim=1).float()
                return itm_logit

            for i, sims in enumerate(tqdm(sims_matrix)):
                topk_sim, topk_idx = torch.topk(sims, k_test)
                image_inputs = vit_feats[i].repeat(k_test, 1, 1).to(model.device)
                score = compute_itm(image_inputs, text_ids[topk_idx], text_atts[topk_idx]).float()
                del image_inputs

                score_matrix_i2t[i, topk_idx] = score + topk_sim
                del topk_sim, topk_idx

            sims_matrix = sims_matrix.t()
            score_matrix_t2i = torch.full(
                (len(texts), len(dataset.image)), -100.0
            ).to(model.device)

            for i, sims in enumerate(tqdm(sims_matrix)):
                topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
                image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
                score = compute_itm(image_inputs, text_ids[i].repeat(k_test, 1), text_atts[i].repeat(k_test, 1))

                score_matrix_t2i[i, topk_idx] = score + topk_sim

            return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    def save_results(self, results, filename):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)