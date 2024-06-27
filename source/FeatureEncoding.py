from torch import nn

class FeatureEncoding(nn.Module):
    def __init__(self):
        super(FeatureEncoding, self).__init__()

        self.rnn_units = 512
        dropout = 0.2

        self.rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        self.rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 


    def forward(self, mask, image, image_mask, audio, audio_mask):
        rnn_img_encoded, (hid, ct) = self.rnn_img(image)
        self.rnn_img_encoded = self.rnn_img_drop_norm(rnn_img_encoded)
        rnn_audio_encoded, (hid_audio, ct_audio) = self.rnn_audio(audio)
        self.rnn_audio_encoded = self.rnn_audio_drop_norm(rnn_audio_encoded)
        
        extended_attention_mask = mask.float().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        self.extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        extended_audio_attention_mask = audio_mask.float().unsqueeze(1).unsqueeze(2)
        extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        self.extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
      
        extended_image_attention_mask = image_mask.float().unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = extended_image_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        self.extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
