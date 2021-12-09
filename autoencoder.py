from torch import nn

class Encoder(nn.Module):
    """
    Encoder class. Learns an embedding space to encode the input.
    """
    
    def __init__(self, nFeatures, embeddingDim=64):
        super(Encoder, self).__init__()
        
        # Create two LSTM layers
        self.layer1 = nn.LSTM(input_size=nFeatures, hidden_size=2 * embeddingDim,
                              num_layers = 1, batch_first=True, dropout=0.25)
        
        self.layer2 = nn.LSTM(input_size=2 * embeddingDim, hidden_size=embeddingDim,
                              num_layers=1, batch_first=True, dropout=0.25)
        
        return
    
    def forward(self, x):
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)
        return x
    
    
class Decoder(nn.Module):
    """
        Decoder Class. Tries to reconstruct the input.
    """
    def __init__(self, inputDim=64, nFeatures=1):
        super(Decoder, self).__init__()
        
        # Capture class variables
        self.nFeatures = nFeatures
        
        # Create two LSTM layers
        self.layer1 = nn.LSTM(input_size=inputDim, hidden_size=inputDim,
                              num_layers = 1, batch_first=True, dropout=0.25)
        
        self.layer2 = nn.LSTM(input_size=inputDim, hidden_size=2*inputDim,
                              num_layers=1, batch_first=True, dropout=0.25)
        
        # Final output layer
        self.out = nn.Linear(2*inputDim, self.nFeatures)
        
        return
    
    def forward(self, x):
        x, _ = self.layer1(x)
        x, _ = self.layer2(x)
        return self.out(x)
    
class AutoEncoder(nn.Module):
    """
        LSTM autoencoder that combines the encoding and decoding into one convenient class.
    """
    
    def __init__(self, nFeatures, embeddingDim=64):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(nFeatures, embeddingDim)
        self.decoder = Decoder(embeddingDim, nFeatures)
        return
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
