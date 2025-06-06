from transformers import Wav2Vec2Model,Wav2Vec2Config,Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutput
from torchaudio.functional import resample
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
W2V_MODEL_PATH = "/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/qiyuan/code/AMD_linear/ckpt/wav2vec"
def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)
class Wav2Vec2ModelLerp(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class AudioProcessor:
    def __init__(
        self,
        device,
        sampling_rate:int=16000,
        only_last_features:bool=False
    ):  
        self._only_last_features = only_last_features
        self.audio_encoder = Wav2Vec2ModelLerp.from_pretrained(W2V_MODEL_PATH, local_files_only=True).to(device)
        self.audio_encoder.requires_grad_(False)
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_MODEL_PATH, local_files_only=True)
        self.sr = sampling_rate
        self.device = device

    @torch.no_grad()
    def process(self, audio:torch.Tensor,sampling_rate:int,seq_len:int):
        if len(audio.shape) == 2:
            audio = torch.mean(audio,dim=0)
        audio = resample(audio,sampling_rate,self.sr)
        audio_feature = np.squeeze(self._processor(audio,sampling_rate=self.sr).input_values)
        audio_feature = torch.from_numpy(audio_feature)
        audio_feature = audio_feature.unsqueeze(0).to(self.device)
        embeddings = self.audio_encoder(audio_feature, seq_len, output_hidden_states=True)
        if self._only_last_features:
            return embeddings.last_hidden_state
        else:
            return torch.concat(embeddings.hidden_states[1:],dim=0).permute(1,0,2)
    @torch.no_grad()
    def __call__(self, audio:torch.Tensor,sampling_rate:int,seq_len:int):
        if len(audio.shape) == 2:
            audio = torch.mean(audio,dim=0)
        audio = resample(audio,sampling_rate,self.sr)
        audio_feature = np.squeeze(self._processor(audio,sampling_rate=self.sr).input_values)
        audio_feature = torch.from_numpy(audio_feature)
        audio_feature = audio_feature.unsqueeze(0).to(self.device)
        embeddings = self.audio_encoder(audio_feature, seq_len, output_hidden_states=True)
        if self._only_last_features:
            return embeddings.last_hidden_state
        else:
            return torch.concat(embeddings.hidden_states[1:],dim=0).permute(1,0,2)
if __name__ == "__main__":
    processor = AudioProcessor(torch.device("cuda:7"),only_last_features=False)
    x = torch.load("/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/liuhuaize/dataset/hdtf_dataset//audio_emb/RD_Radio11_000.pt")
    # hidden_states = processor.process(x,44100,40)
    # print(hidden_states.shape)