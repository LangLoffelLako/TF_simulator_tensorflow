# %% [markdown]
# # Imports

# %%
# logging and decorators
import logging as log
import time

# system tools
import pathlib

# general modules
import numpy as np

# tensorflow modules
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras import layers
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

# necessary for visualization and user input
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %% [markdown]
# # Settings

# %%
# logging settings
log.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s', 
    # log.INFO for normal run
    level=log.INFO,
    # log.DEBUG for diagnostics
    # level=log.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
log_enabled = True

# Set True, if code is run as jupyter notebook
is_interactive_notebook = True

# paths
corpus_file_path = 'datasets\\corpus'
bco_file_path = "datasets\\bookscorpusopen\\epubtxt"
tight_fit_512_dataset_path = 'datasets\\tight_fit'
vocab_path = 'datasets\\vocab.txt'

# tokenizer
tokenizer_name = 'story_corpus_tokenizer'
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

# %% [markdown]
# # Decorators

# %% [markdown]
# # Architecture

# %% [markdown]
# ## Helper functions

# %%
def do_nothing(*args, **kwargs):
    pass

def clones(layer_class, N, **kwargs):
    """Produce N identical layers"""
    return [layer_class(**kwargs) for layer_number in range(N)]

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0


# %% [markdown]
# ### Layer Wrapper

# %%
class VisualWrapper():
    should_visualize = False
    instances = []

    def __init__(self, vis_on_count=None, enabler=False):
        self.counter = 0
        self.vis_on_count = vis_on_count if vis_on_count else []
        self.enabler = enabler
        VisualWrapper.instances.append(self)

    def visualize_data(self, data_x, mode, training, text=None, data_y=None, vis_diff=False):
        # TODO: data_y and vis_diff are there to be used for calculating the 
        if training is False:
            # check for visualisation param of the instance and visualize or change class settings
            if self.counter in self.vis_on_count:  
                if self.should_visualize:   
                    tf.print(text)  
                    self.choose_func(mode)(data_x)
                if self.enabler:
                    VisualWrapper.should_visualize = True
            else:
                if self.enabler:
                    VisualWrapper.should_visualize = False
            self.counter += 1

    # @log_dec
    def choose_func(self, mode):
        if mode == 'color_bar':
            return lambda x: self.color_bar(x)
        elif mode == 'print':
            return lambda x: self.print_data(x)
        elif mode == 'reduce_dim':
            return lambda x: self.reduce_dim(x)
        else:
            return do_nothing

    def color_bar(self, tensor):
        x_label = 'Positions'
        y_label = 'Embbeddings'
        # Assuming data[0] is a numpy array.
        # If it's a ListWrapper or another list-like object, convert it to a numpy array.
        tensor = np.array(tensor[0])
        # If the array is 1D, reshape it into a 2D array with one column
        if tensor.ndim == 1:
            tensor = np.reshape(tensor, (-1, 1))
        # Set the size of the plot (you can adjust the dimensions as needed)
        plt.figure(figsize=(10, 2))
        # Use imshow to create a color-coded visualization
        plt.imshow(tensor, cmap='jet', aspect='auto')
        plt.colorbar(label='Tensor Value')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        plt.show()

    def print_data(self,data):
        tf.print(data)

    def reduce_dim(self, tensor):

        array = np.squeeze(tensor, axis=0)

        # array = tensor.numpy().reshape(tensor.shape[0], -1)

        scaled_array = array / np.min(np.abs(array))

        # tf.print('scaled_numpy_array', scaled_array)

        # TODO: PCA must be trained
        # Other algorithms could be tsne or umap
        pca = PCA(n_components=3)
        reduced_array = pca.fit_transform(scaled_array)

        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(3, 3))
        
        ax = fig.add_subplot(111, projection='3d')

        # Create a scatter plot of the UMAP embeddings
        # Color each point by the sample's label
        # scatter = ax.scatter(reduced_array[:, 0], reduced_array[:, 1], reduced_array[:, 2], s=50)
        # Optional: include a colorbar if you have labels
        # plt.colorbar(scatter)

        # Create a quiver plot to visualize each point as a vector from the origin
        ax.quiver(0, 0, 0, reduced_array[:, 0], reduced_array[:, 1], reduced_array[:, 2], arrow_length_ratio=0.1)
       # ax.quiver(0, 0, reduced_array[:, 0], reduced_array[:, 1], arrow_length_ratio=0.1)

        # Add some helpful labels
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title('Embeddings')

        boundaries = np.max(reduced_array)

        # Set the x and y axis limits
        ax.set_xlim([-boundaries, boundaries])
        ax.set_ylim([-boundaries, boundaries])
        ax.set_zlim([-boundaries, boundaries])


        # Set aspect of the plot to equal to ensure that the vectors are displayed correctly
        # ax.set_aspect('equal')

        # Show the plot
        plt.show()

    @classmethod
    def reset_counter(cls):
        for instance in cls.instances:
            instance.counter = 0

# %% [markdown]
# ## Main Layers

# %% [markdown]
# These classes are built using the Keras Functional API, which provides more flexibility than the Sequential API for defining complex models. Each class is a subclass of tf.keras.layers.Layer, so they can be composed to build more complex layers or models. The call method of each class defines the computation that the layer performs.
# 
# These classes are designed to be components of a larger transformer model. The model itself is typically composed of an encoder and a decoder, each of which is made up of a stack of identical layers. The layers themselves contain sublayers that perform operations such as self-attention, source attention (in the case of the decoder), and position-wise feed-forward networks. These operations are encapsulated within classes like `EncoderStack`, `DecoderStack`, `EncoderLayer`, `DecoderLayer`, and `PositionwiseFeedForward`. The layer norm and dropout are applied in `ResidualSublayer` for regularizing and speeding up the training process.

# %% [markdown]
# ### Encoder Decoder Layer

# %% [markdown]
# 1. `EncoderDecoder`:
#     - `__init__(self, encoder, decoder, enc_embed, dec_embed, generator)`: This initializes the EncoderDecoder instance. It takes in five arguments:
#         - `encoder`: The encoder layer to be used.
#         - `decoder`: The decoder layer to be used.
#         - `enc_embed`: The embedding layer for the encoder.
#         - `dec_embed`: The embedding layer for the decoder.
#         - `generator`: The final layer that generates the output tokens.
#     - `encode(self, inputs, pad_mask)`: This method is used to encode the inputs using the encoder layer. It takes in two arguments:
#         - `inputs`: The input tokens to be encoded.
#         - `pad_mask`: The mask indicating which tokens are padding.
#     - `decode(self, enc_input, pad_mask, inputs, subseq_mask)`: This method is used to decode the encoded inputs using the decoder layer. It takes in four arguments:
#         - `enc_input`: The encoded input from the encoder.
#         - `pad_mask`: The mask indicating which tokens are padding in the encoded input.
#         - `inputs`: The target tokens to be decoded.
#         - `subseq_mask`: The mask indicating which tokens in the target sequence should not be attended to.
#     - `call(self, enc_input, dec_input, pad_mask, subseq_mask)`: This method is used to perform the complete transformation from input tokens to output tokens. It takes in four arguments that are the same as those described in the `encode` and `decode` methods.

# %%
# @LayerWrapperDecorator(visualize_on_calls=[1], visual_setter=True)
class EncoderDecoder(tf.keras.Model, VisualWrapper):
    def __init__(self, encoder_stack, decoder_stack, enc_embed, dec_embed, generator):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0], enabler=False)
        # modules
        self.encoder_stack = encoder_stack
        self.decoder_stack = decoder_stack
        self.enc_embed = enc_embed
        self.dec_embed = dec_embed
        self.generator = generator

    # @log_dec
    def encode(self, inputs, pad_mask, training=None):
        return self.encoder_stack(self.enc_embed(inputs), pad_mask, training=training)
    
    # @log_dec
    def decode(self, enc_input, pad_mask, inputs, subseq_mask, training=None):
        return self.decoder_stack(self.dec_embed(inputs), enc_input, pad_mask, subseq_mask, training=training)

    # @log_dec
    def call(self, inputs, training=None):

        enc_input, dec_input, pad_mask, subseq_mask = inputs

        if not training: # Additional training = False check, such that calculations for execution are not conducted unless not training
            input_emb_enc = self.enc_embed(enc_input)
            input_emb_dec = self.dec_embed(dec_input)
            self.visualize_data(input_emb_enc- input_emb_dec, mode='color_bar', training=training, text='The difference between the enc_emb and the dec_emb')

        return self.decode(self.encode(enc_input, pad_mask, training), 
                           pad_mask,
                           dec_input, 
                           subseq_mask, training)

# %% [markdown]
# ### Layer Norm

# %% [markdown]
# 
# 2. `LayerNorm`:
#     - `__init__(self, features, eps=1e-6)`: This initializes the LayerNorm instance. It takes in two arguments:
#         - `features`: The number of features in the input to be normalized.
#         - `eps`: A small number to add to the denominator for numerical stability.
#     - `call(self, x)`: This method is used to apply layer normalization to the input. It takes in one argument:
#         - `x`: The input to be normalized.

# %%
class LayerNorm(layers.Layer, VisualWrapper):

    def __init__(self, features, eps=1e-6) -> None:
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.a_2 = self.add_weight(shape=(features,), initializer='ones', name=self.name + "a_2")
        self.b_2 = self.add_weight(shape=(features,), initializer='zeros', name=self.name + "b_2")
        self.eps = eps

    # @log_dec
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
        std = tf.math.sqrt(var + self.eps)
        return self.a_2 * (x - mean) / std + self.b_2

# %% [markdown]
# ### Residual Layer

# %% [markdown]
# 
# 3. `ResidualSublayer`:
#     - `__init__(self, size, dropout)`: This initializes the ResidualSublayer instance. It takes in two arguments:
#         - `size`: The number of features in the input.
#         - `dropout`: The dropout rate to be applied after the sublayer.
#     - `call(self, x, sublayer)`: This method is used to apply a sublayer and a residual connection to the input. It takes in two arguments:
#         - `x`: The input to be transformed.
#         - `sublayer`: The sublayer to be applied to the input. This is expected to be a function or callable object that takes in the input and returns a tensor of the same shape.

# %%
class ResidualSublayer(layers.Layer, VisualWrapper):
    """
    A residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout) -> None:
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.norm = LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    # @log_dec
    def call(self, x, sublayer, training=None):
        "Apply residual connection to any sublayer with the same size."
        sublayer_out = sublayer(self.norm(x))
        return x + self.dropout(sublayer_out, training=training)

# %% [markdown]
# ### Encoder Stack Layer

# %% [markdown]
# 4. `EncoderStack`:
#     - `__init__(self, layer, N)`: This initializes the EncoderStack instance. It takes in two arguments and initializes two instance variables:
#         - `layer`: The type of layer to be used in the encoder stack. This should be a callable object that takes in the input and a mask and returns a tensor.
#         - `N`: The number of layers in the encoder stack.
#         - `self.layers` is a list of `N` layer clones of the type `layer`.
#         - `self.norm` is the norm layer, that is applied to the output of the `EncoderStack`.
#     - `call(self, x, mask)`: This method is used to pass the input through each layer in the encoder stack in turn. It takes in two arguments:
#         - `x`: The input to be processed by the encoder stack.
#         - `mask`: The mask indicating which tokens should not be attended to.

# %%
class EncoderStack(layers.Layer, VisualWrapper):
    """
    Core encoder is a stack of N=6 Layers
    """

    def __init__(self, layer, N, size, **kwargs):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.layers = clones(layer, N, size=size, **kwargs)
        self.norm = LayerNorm(size)

    # @log_dec
    def call(self, x, mask, training=None):
        """
        Pass the input (and mask) through each layer in turn
        """
        for layer in self.layers:
            x = layer(x, mask, training=training)
        return self.norm(x)

# %% [markdown]
# ### Encoder Layer

# %% [markdown]
# 
# 5. `EncoderLayer`:
#     - `__init__(self, size, self_attn, feed_forward, dropout)`: This initializes the EncoderLayer instance. It takes in four arguments:
#         - `size`: The number of features in the input.
#         - `self_attn`: The self-attention mechanism to be used in the encoder layer. This should be a callable object that takes in the input and a mask and returns a tensor.
#         - `feed_forward`: The feed-forward network to be used in the encoder layer. This should be a callable object that takes in the input and returns a tensor.
#         - `dropout`: The dropout rate to be applied after each sublayer.
#     - `call(self, x, mask)`: This method is used to pass the input through the self-attention mechanism and the feed-forward network. It takes in two arguments:
#         - `x`: The input to be processed by the encoder layer.
#         - `mask`: The mask indicating which tokens should not be attended to.

# %%
class EncoderLayer(layers.Layer, VisualWrapper):
    """
    Encoder is made up of a self-attention and a feed forward layer 
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSublayer, N=2, size=size, dropout=dropout)

    # @log_dec
    def call(self, x, mask, training=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, training=training), training=training)
        return self.sublayer[1](x, lambda x: self.feed_forward(x, training=training), training=training)

# %% [markdown]
# ### Decoder Stack Layer

# %% [markdown]
# 6. `DecoderStack`:
#     - `__init__(self, layer, N)`: This initializes the DecoderStack instance. It takes in two arguments and initializes two instance variables:
#         - `layer`: The type of layer to be used in the decoder stack. This should be a callable object that takes in the input, the memory from the encoder, a source mask, and a target mask, and returns a tensor.
#         - `N`: The number of layers in the decoder stack.
#         - `self.layers` is a list of `N` layer clones of the type `layer`.
#         - `self.norm` is the norm layer, that is applied to the output of the `EncoderStack`.
#     - `call(self, x, memory, src_mask, tgt_mask)`: This method is used to pass the input through each layer in the decoder stack in turn. It takes in four arguments:
#         - `x`: The input to be processed by the decoder stack.
#         - `memory`: The output of the encoder, which serves as the memory for the decoder.
#         - `src_mask`: The mask indicating which tokens in the source sequence should not be attended to.
#         - `tgt_mask`: The mask indicating which tokens in the target sequence should not be attended to.

# %%
class DecoderStack(layers.Layer, VisualWrapper):
    """
    Generic N layer decoder with masking
    """

    def __init__(self, layer, N, size, **kwargs):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.layers = clones(layer, N, size=size, **kwargs)
        self.norm = LayerNorm(size)

    # @log_dec
    def call(self, x, memory, src_mask, tgt_mask, training=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, training=training)
        return self.norm(x)

# %% [markdown]
# ### Decoder Layer

# %% [markdown]
# 7. `DecoderLayer`:
#     - `__init__(self, size, self_attn, src_attn, feed_forward, dropout)`: This initializes the DecoderLayer instance. It takes in five arguments:
#         - `size`: The number of features in the input.
#         - `self_attn`: The self-attention mechanism to be used in the decoder layer. This should be a callable object that takes in the input and a mask and returns a tensor.
#         - `src_attn`: The source attention mechanism to be used in the decoder layer. This should be a callable object that takes in the input, the memory from the encoder, and a mask, and returns a tensor.
#         - `feed_forward`: The feed-forward network to be used in the decoder layer. This should be a callable object that takes in the input and returns a tensor.
#         - `dropout`: The dropout rate to be applied after each sublayer.
#     - `call(self, x, memory, src_mask, tgt_mask)`: This method is used to pass the input through the self-attention mechanism, the source attention mechanism, and the feed-forward network. It takes in four arguments:
#         - `x`: The input to be processed by the decoder layer.
#         - `memory`: The output of the encoder, which serves as the memory for the decoder.
#         - `src_mask`: The mask indicating which tokens in the source sequence should not be attended to.
#         - `tgt_mask`: The mask indicating which tokens in the target sequence should not be attended to.

# %%
class DecoderLayer(layers.Layer, VisualWrapper):
    """
    Decoder is made of self-attn, source-attn and feedforward layer
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSublayer, N=3, size=size, dropout=dropout)

    # @log_dec
    def call(self, x, memory, src_mask, tgt_mask, training=None):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# %% [markdown]
# ## Sublayers

# %% [markdown]
# ### Feedforward Layer

# %% [markdown]
# 8. `PositionwiseFeedForward`:
#     - `__init__(self, d_model, d_ff, dropout=0.1, *args, **kwargs)`: This initializes the PositionwiseFeedForward instance. It takes in three arguments and an optional set of arguments:
#         - `d_model`: The number of features in the input.
#         - `d_ff`: The number of features in the hidden layer of the feed-forward network.
#         - `dropout`: The dropout rate to be applied after the first layer of the feed-forward network.
#         - `*args, **kwargs`: Additional arguments that might be necessary for the parent class initialization.
#     - `call(self, x)`: This method is used to pass the input through the feed-forward network. It takes in one argument:
#         - `x`: The input to be processed by the feed-forward network.

# %%
class PositionwiseFeedForward(layers.Layer, VisualWrapper):
    """Implements FFN equation"""

    def __init__(self, d_model, d_ff, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.w_1 = layers.Dense(d_ff)
        self.w_2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    # @log_dec
    def call(self, x, training=None):
        return self.w_2(self.dropout(tf.nn.relu(self.w_1(x)), training=training))

# %% [markdown]
# ### Generator Layer

# %% [markdown]
# 9. `Generator`:
#     - `__init__(self, vocab)`: This method initializes the Generator instance. It accepts one argument:
#         - `vocab`: The size of the vocabulary which will be the number of output units in the dense layer.
#     - `call(self, x)`: This method is used to pass the input through the generator. It takes in one argument:
#         - `x`: The input tensor to be processed by the generator. The method returns the log softmax of the output of the dense layer.

# %%
# @LayerWrapperDecorator(visualize_on_calls=[1], visualizations=[('mode1', 'x')])
class Generator(layers.Layer, VisualWrapper):
    """
    Define standard linear + softmax generation step
    """

    def __init__(self, vocab):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.proj = layers.Dense(vocab)

    # @log_dec
    def call(self, x, training=None):
        result = tf.nn.log_softmax(self.proj(x), axis=-1)
        self.visualize_data(result, 
                            'color_bar', 
                            text=f"This is the data from {self.__class__.__name__}", 
                            training=training)
        return result

# %% [markdown]
# ### Attention Layer

# %% [markdown]
# 
# 10. `attention(query, key, value, mask=None, dropout=None)`:
#     - This is a function that computes the 'Scaled Dot Product Attention'. The arguments are as follows:
#         - `query`, `key`, `value`: These are the main inputs to the attention function.
#         - `mask`: Optional mask for the attention scores.
#         - `dropout`: Optional dropout rate to be applied to the attention scores.
#     - The function first scales the dot product of the query and key, applies the mask if provided, applies softmax to compute attention scores, applies dropout if provided, and then uses the attention scores to compute a weighted sum of the value inputs.

# %%
# @log_dec
def attention(query, key, value, mask=None, dropout=None, training=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.shape[-1]
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / tf.sqrt(d_k)
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.bool)
        scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))
    p_attn = tf.nn.softmax(scores, axis=-1)
    if dropout is not None:
        p_attn = dropout(p_attn, training=training)
    return tf.matmul(p_attn, value), p_attn

# %% [markdown]
# 11. `MultiHeadedAttention`:
#     - `__init__(self, h, d_model, dropout=0.1)`: This initializes the MultiHeadedAttention instance. It takes in three arguments:
#         - `h`: The number of attention heads.
#         - `d_model`: The number of features in the input.
#         - `dropout`: The dropout rate to be applied after the softmax in the attention computation.
#     - `call(self, query, key, value, mask=None)`: This method is used to compute the multi-headed attention over the inputs. It takes in four arguments:
#         - `query`, `key`, `value`: These are the main inputs to the attention computation.
#         - `mask`: Optional mask for the attention scores.
#     - The method first computes the linear projections of the inputs, applies the attention function to the projected inputs, concatenates the outputs of the attention function across the attention heads, and then applies a final linear transformation to the concatenated outputs.

# %%
class MultiHeadedAttention(layers.Layer, VisualWrapper):
    
    def __init__(self, h, d_model, dropout=0.1, **kwargs):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.query, self.key, self.value, self.linear = clones(layers.Dense, N=4, units=d_model)
        self.attn = None
        self.dropout = layers.Dropout(dropout)

    # @log_dec
    def call(self, query, key, value, mask=None, training=None):
        if mask is not None:
            # Same mask applied to all h heads
            mask = tf.expand_dims(mask, 1)

        nbatches = tf.shape(query)[0]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            tf.transpose(tf.reshape(lin(x), [nbatches, -1 , self.h, self.d_k]), perm=[0, 2, 1, 3]) 
            for lin, x in zip(
                [self.query, self.key, self.value], 
                (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, training=training)

        # 3) "Concat" using a view and apply a final linear.
        x = tf.reshape(tf.transpose(x ,perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))

        return self.linear(x)

# %% [markdown]
# ### Positional Embedding Layer

# %% [markdown]
# 12. `positional_encoding(length, depth)`:
#     - This is a function that computes the positional encoding for a sequence of a given length and depth. The arguments are as follows:
#         - `length`: The length of the sequence for which positional encoding is to be computed.
#         - `depth`: The number of features in the input sequence.
#     - The function first computes the rates at which the angles should change across the positions and depths, then computes the angles at each position and depth, and finally applies sine to the angles at the even indices and cosine to the angles at the odd indices. The positional encoding for a position is thus a vector of these sine and cosine values.

# %%
# @log_dec
def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]   # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)

    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads  = positions * angle_rates           # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
        )

    return tf.cast(pos_encoding, dtype=tf.float32)

# %% [markdown]
# 13. `PositionalEmbedding`:
#     - `__init__(self, vocab_size, d_model)`: This method initializes the PositionalEmbedding instance. It takes in two arguments:
#         - `vocab_size`: The size of the vocabulary, which will be the input dimension of the embedding layer.
#         - `d_model`: The number of features to be output by the embedding layer and the depth for the positional encoding.
#     - `call(self, x)`: This method is used to compute the positionally encoded embeddings of the inputs. It takes in one argument:
#         - `x`: The input tensor for which the embeddings are to be computed.
#     - The method first computes the embeddings of the inputs, scales the embeddings by the square root of `d_model`, and then adds the positional encoding to these scaled embeddings.

# %%
class PositionalEmbedding(layers.Layer, VisualWrapper):
    def __init__(self, vocab_size, d_model, dropout):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0,1,2])
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.dropout = layers.Dropout(dropout)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    # @log_dec
    # @tf.function
    def call(self, x, training=None):

        length = tf.shape(x)[1]
        x_emb = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding
        x_emb_scale = x_emb * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        y = self.dropout(x_emb_scale + self.pos_encoding[tf.newaxis, :length, :])

        self.visualize_data(x_emb, mode='color_bar', text=f"This is the embedding of the input to {self.__class__.__name__}.", training=training)
        self.visualize_data(y, mode='color_bar', text=f"This is the embedding of the input to {self.__class__.__name__} with added positional encoding.", training=training)
        self.visualize_data(x_emb-y, mode='color_bar', text=f"Here you can see the difference between both.", training=training)
        return y

# %% [markdown]
# ## Model Generation

# %% [markdown]
# ### Make model

# %% [markdown]
# 14. `make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)`:
#     - The `make_model` function constructs a Transformer model from given hyperparameters. It takes seven arguments:
#         - `src_vocab`: The size of the source vocabulary.
#         - `tgt_vocab`: The size of the target vocabulary.
#         - `N`(default=6): The number of layers in the Transformer's Encoder and Decoder stacks.
#         - `d_model`(default=512): The dimension of the model. It's the number of features in input and output.
#         - `d_ff`(default=2048): The number of features in the hidden layer of the feed-forward network.
#         - `h`(default=8): The number of attention heads in the MultiHeadedAttention mechanism.
#         - `dropout`(default=0.1): The dropout rate to be applied in several parts of the model.
#     - Inside this function, instances of `MultiHeadedAttention` and `PositionwiseFeedForward` are created. These instances are then deep-copied and used to construct the Encoder and Decoder stacks, additionally the PositionalEmbeddings, and the Generator are instantiated. All these parts are then assembled into a `EncoderDecoder` instance, which includes the complete Transformer model. If a module is wrapped with a `LayerWrapper` this is in order to visualize the output of this layer on sucessive calls. Look for the specific meaning of the `LayerWrapper` parameters in the definition of the class.
#     - Finally, the function returns the constructed model. 

# %%
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1) -> tf.keras.Model:
    "Helper: Construct a model from hyperparameters."
    
    model = EncoderDecoder(
                EncoderStack(
                    EncoderLayer,
                    N=N, 
                    size=d_model, 
                    dropout=dropout, 
                    self_attn=MultiHeadedAttention(h, d_model), 
                    feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout)),
                DecoderStack(
                    DecoderLayer, 
                    N=N, 
                    size=d_model, 
                    dropout=dropout,
                    self_attn=MultiHeadedAttention(h, d_model), 
                    src_attn=MultiHeadedAttention(h, d_model), 
                    feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout)),
                PositionalEmbedding(
                    src_vocab, 
                    d_model,
                    dropout),
                PositionalEmbedding(
                    tgt_vocab, 
                    d_model,
                    dropout),
                Generator(tgt_vocab)
            )

    return model

# %% [markdown]
# # Training

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Tokenizer

# %%
class StoryTokenizer(tf.Module, VisualWrapper):
    def __init__(self, reserved_tokens, vocab_path):    
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)        

    def tokenize(self, strings, training=None):

        encoded = self.tokenizer.tokenize(strings)
        merged_enc = encoded.merge_dims(-2, -1)
        out = self.add_start_end(merged_enc)

        self.visualize_data(self.lookup(out),
                            mode='print', 
                            text=f"This is the data from {self.__class__.__name__}", 
                            training=training)

        return out
    
    def detokenize(self, tokenized, training=None):
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(self._reserved_tokens, words)
    
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @staticmethod
    def add_start_end(ragged):
        START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
        END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

        count = ragged.bounding_shape()[0]
        starts = tf.fill([count, 1], START)
        ends = tf.fill([count, 1], END)
        return tf.concat([starts, ragged, ends], axis=1)

    @staticmethod
    def cleanup_text(reserved_tokens, token_txt):
        bad_tokens = list(filter(lambda token: token != "[UNK]", reserved_tokens))
        bad_tokens_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_tokens_re)
        ragged_result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        result = tf.strings.reduce_join(ragged_result, separator=' ', axis=-1)

        return result
    
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]
    
    def get_vocab_path(self):
        return self._vocab_path
    
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

# %% [markdown]
# ### Vocabulary Generation

# %%
def load_dataset(dataset_text_file):
    return tf.data.TextLineDataset(filenames=dataset_text_file)

def create_vocab(dataset):
    bert_vocab_args=dict(
        vocab_size = 8000,
        reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"],
        bert_tokenizer_params = dict(lower_case=True),
        learn_params = {},
    )

    story_vocab = bert_vocab.bert_vocab_from_dataset(
        dataset.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    return story_vocab

def create_vocab_from_textdata(text_file=corpus_file_path):
    dataset = load_dataset(text_file)
    vocab = create_vocab(dataset)
    return vocab

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as file:
        for token in vocab:
            print(token, file=file)

# %% [markdown]
# ### Dataset generator

# %%
class DatasetGenerator():

    def __init__(self,
                 tokenizer, 
                 buffer_size=20000, 
                 batch_size=64, 
                 max_padding=128, 
                 pad_id=0):
        
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_padding = max_padding
        self.pad_id = pad_id
        self.dataset = None

    def txt_files_to_lines_gen(self, file_path):
        path = pathlib.Path(file_path)
        for file in path.iterdir():
            if file.is_file():
                with open(file, 'r') as f:
                    for line in f:
                        yield line.strip()

    def lines_to_fit_sentences(self, sentences, length):
        length = length / 1.5 # estimate of token/word ratio (in real the value is about 1.4)

        current_combined_sentence = ""

        for sentence in sentences:
            sentence = sentence.strip()  # Remove leading/trailing whitespace
            sentence_words = sentence.split()

            # Check if combining the current sentence with the previous one exceeds the word limit
            if len(current_combined_sentence.split()) + len(sentence_words) > length:
                yield current_combined_sentence
                current_combined_sentence = sentence  # Start a new combined sentence
            else:
                current_combined_sentence += " " + sentence  # Concatenate the sentences
    
    def generate_datasets(self, file_path):
        
        # Create a Dataset from the text file
        lines_gen = self.txt_files_to_lines_gen(file_path)
        fit_sentence_gen = self.lines_to_fit_sentences(lines_gen, 512)
        
        dataset = tf.data.Dataset.from_generator(lambda: fit_sentence_gen, 
                                                 output_signature=tf.TensorSpec(shape=(), 
                                                                                dtype=tf.string))

        # Tokenize the whole dataset with the pre-trained tokenizer
        tokenized_dataset = (dataset
                            .shuffle(self.buffer_size)
                            .batch(self.batch_size)
                            .map(lambda x: self.prepare_datapoint(x))
                            .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        self.dataset = tokenized_dataset
        
        return tokenized_dataset
    
    def prepare_datapoint(self, data_point):
        
        src_tokens = self.tokenizer.tokenize(data_point)
        tgt_tokens = src_tokens[:, :-1]
        label_tokens = src_tokens[:, 1:]

        src = src_tokens.to_tensor(shape=[1, self.max_padding], 
                                   default_value=self.pad_id)
        tgt = tgt_tokens.to_tensor(shape=[1, self.max_padding], 
                                   default_value=self.pad_id)
        label = label_tokens.to_tensor(shape=[1, self.max_padding], 
                                       default_value=self.pad_id)
        
        src_mask = (src != self.pad_id)[:, np.newaxis, :]
        tgt_mask = self.make_std_mask(tgt)

        return (src, tgt, src_mask, tgt_mask), label
  
    def make_std_mask(self, tgt):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != self.pad_id)[:, np.newaxis, :]
        tgt_mask = tf.logical_and(tgt_mask, subsequent_mask(tgt.shape[-1]))
        return tgt_mask

# %% [markdown]
# ## Training

# %% [markdown]
# ### Loss function

# %%
class LabelSmoothingLoss(layers.Layer, VisualWrapper):
    """
    This class represents a loss function layer that applies label smoothing to prevent overconfidence 
    in the model's predictions. This is done by replacing the 0s and 1s in the labels with smoothed values, 
    such that the model learns to be less confident and thus, more robust.

    Args:
        vocab_size (int): The size of the vocabulary, which also represents the number of classes.
        padding_idx (int): The index representing padding elements.
        smoothing (float): The smoothing factor to be applied. The values should be between 0 and 1. 
                           Default value is 0.
        reduction (tf.keras.losses.Reduction): The type of reduction to apply to the output loss. 
                                               Default is tf.keras.losses.Reduction.SUM.

    Methods:
        call(x, target): Calculates and returns the loss given the model's output `x` and the target labels.

    Example:
        >>> loss_func = LabelSmoothingLoss(vocab_size=5000, padding_idx=0, smoothing=0.1)
        >>> x = tf.random.uniform((10, 5000))  # model's output
        >>> target = tf.random.uniform((10, 1), maxval=5000, dtype=tf.int32)  # target labels
        >>> loss = loss_func(x, target)  # calculate loss
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_func = tf.keras.losses.KLDivergence(reduction='none')

    # @log_dec
    def call(self, x, target):
        # create padding mask
        mask = self.padding_mask(target, self.padding_idx)

        # tf.print('loss mask:', mask)

        # Apply label confidence
        true_dist = target * self.confidence

        # Apply label smoothing
        smoothing_value = self.smoothing / tf.cast(self.vocab_size - 2, tf.float32)
        true_dist = tf.where(tf.equal(true_dist, 0), smoothing_value, true_dist)

        # tf.print('prediction before loss:', x)
        # tf.print('one hot smoothed', true_dist)
        # tf.print('smoothed one hot high values at:', tf.where(true_dist > self.smoothing))

        # Calculate the loss
        loss = self.kl_div_loss(x, true_dist)

        # tf.print('loss tensor:', loss)

        loss = tf.cast(self.apply_mask(loss, mask), x.dtype)

        # tf.print('loss after masking:', loss)
        # tf.print('loss big at:', tf.where(loss > 0.1))
        # tf.print('reduced sum loss:', tf.reduce_sum(loss))

        loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

        return loss
    
    @staticmethod
    def padding_mask(t, padding_idx):
        return tf.cast(tf.equal(t[:, :, padding_idx], 0), tf.float32)

    @staticmethod
    def apply_mask(t, mask):
        return t * (tf.reshape(mask, [-1, 1]) * tf.ones_like(t))
    
    @staticmethod
    def kl_div_loss(input, target):
        return target * (tf.math.log(target)-input)

# %%
class LossCompute(tf.keras.losses.Loss, VisualWrapper):
    '''TODO: Correct Loss Computation'''
    def __init__(self, generator, loss_function, vocab_size, name='loss_compute'):
        super().__init__(name=name)
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.generator = generator
        self.loss_function = loss_function
        self.vocab_size = vocab_size

    # @log_dec
    def call(self, y_true, y_pred): 
        # tf.print('initial y_pred', y_pred)
        # tf.print('initial y_true', y_true)
        y_pred = self.generator(y_pred)
        y_true_one_hot = tf.cast(tf.one_hot(y_true, depth=self.vocab_size), tf.float32)
        # Compute loss
        # tf.print('y_true_one_hot', tf.where(y_true_one_hot != 0))
        # tf.print(tf.argmax(y_true_one_hot, axis=-1))
        loss = self.loss_function(y_pred, y_true_one_hot)
        # Calculate mean loss per batch
        norm = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        
        sloss = loss / norm

        # Return scaled loss (mean loss per batch) and total loss (for the whole batch)
        return loss
        # return sloss * norm, sloss

# %% [markdown]
# ### Learning rate and optimizer

# %%
class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule, VisualWrapper):
    def __init__(self, d_model=512, warmup_steps=4000):
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    # @log_dec
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg_1 = tf.math.rsqrt(step)
        arg_2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg_1, arg_2)

# %% [markdown]
# ### Training metrics

# %%
# @log_dec
def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)

  match = label == pred
  mask = label != 0
  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  return tf.reduce_sum(match)/tf.reduce_sum(mask)

# %% [markdown]
# #### Compile and fit model

# %%
def run_model(tokenizer, training_data, validation_data, config):

    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    base_lr = config["base_lr"]
    max_padding = config["max_padding"]
    padding_idx = config["padding_idx"]
    warmup_steps = config["warmup_steps"]
    d_model = config["d_model"]
    N = config["N"]
    h = config["h"]
    fit_verbose = config["fit_verbose"]
    load_latest = config["load_latest"]
    save_model = config["save_model"]

    vocab_size = tokenizer.get_vocab_size()

    model = make_model(vocab_size, vocab_size, d_model=d_model, N=N, h=h)

    # out = model.decode(model.encode(first_batch.src, first_batch.src_mask), 
    #                       first_batch.src_mask,
    #                       first_batch.tgt, 
    #                       first_batch.tgt_mask)

    # out = model(first_batch.src, first_batch.tgt, first_batch.src_mask, first_batch.tgt_mask)

    model.compile(
        loss = LossCompute(model.generator, 
                           LabelSmoothingLoss(vocab_size, padding_idx=padding_idx, smoothing=0.1), 
                           vocab_size=vocab_size), 
        optimizer = tf.keras.optimizers.Adam(TransformerSchedule(d_model=d_model, 
                                                                 warmup_steps=warmup_steps), # type: ignore
                                                                 beta_1=0.9, 
                                                                 beta_2=0.98, 
                                                                 epsilon=1e-9), 
        metrics = [masked_accuracy],
    )

    model = load_latest_weights(model, d_model, load_latest=load_latest)

    if save_model:

        current_time = time.strftime("%Y%m%d-%H%M%S")

        directory = f"model_N{N}_h{h}_d{d_model}_t{current_time}"
        ckp_name = "model_{epoch:03d}.h5"
        final_name = "final_model.h5"
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = dir_path / ckp_name
        final_path = dir_path / final_name
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_freq='epoch', 
            save_weights_only=True, 
            verbose=1)
    else:
        checkpoint = []

    if n_epochs > 0:
        # TODO: Return to fullsized dataset
        model.fit(training_data,
            epochs = n_epochs,
            batch_size = batch_size,
            validation_data = validation_data,
            callbacks = [checkpoint],
            verbose = fit_verbose)
    
    if save_model:
        model.save_weights(final_path, overwrite=True)

    print(model.summary())

    VisualWrapper.reset_counter()
    
    return model

def load_latest_weights(model, d_model, load_latest=False, model_folder=None):
    # TODO: Ensure architecture sizes match.
    if model_folder is not None:
        # Load weights from the specified model folder
        directories = [model_folder]
    elif load_latest:
        directories = sorted(pathlib.Path('.').glob('model_N*_h*'), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        return model

    # Load weights from the latest trained model
    latest_weights = None
    if directories:
        latest_dir_path = directories[0]
        # Get all the h5 files inside the directory and sort them
        h5_files = sorted(latest_dir_path.glob('*.h5'))

        if h5_files:
            # Pick the last epoch file (or final_model file if it exists)
            latest_epoch_file = h5_files[-1] if 'final_model.h5' not in str(h5_files[-1]) else h5_files[-2]
            latest_weights = latest_epoch_file

    # Load weights if we found a previously trained model
    if latest_weights is not None:
        print(f'Loading weights from {latest_weights}')
        
        # Create a dummy input matching the input shape of the model
        dummy_input = tf.random.uniform(shape=[1,d_model]), tf.random.uniform(shape=[1,d_model]), None, None
        # Call the model on the dummy input
        _ = model.generator(model(dummy_input))

        model.load_weights(latest_weights)
    return model

# %% [markdown]
# ### Model configuration

# %%
config = {
        "batch_size": 64,
        "n_epochs": 1,
        "base_lr": 1.0,
        "max_padding": 512,
        "padding_idx": 0,
        "warmup_steps": 1000,
        "N": 6,
        "d_model": 512,
        "h": 8,
        "fit_verbose": 1,
        "load_latest": False,
        "save_model": True,
    }

# %% [markdown]
# ### Load dataset and tokenizer

bco_generator = DatasetGenerator(StoryTokenizer(reserved_tokens, vocab_path),
                                 batch_size=config["batch_size"], 
                                 max_padding=config["max_padding"], 
                                 pad_id=config["padding_idx"])

bco_dataset = bco_generator.generate_datasets(bco_file_path)

# %%
bco_dataset.save("tfds_shards")

# %% [markdown]
# ### Run model training

# %%
model = run_model(StoryTokenizer(reserved_tokens, vocab_path), 
                  bco_dataset.take(900000), 
                  bco_dataset.skip(900000).take(300000),
                  config)

# %% [markdown]
# # Inference

# %% [markdown]
# ## WordComplete model

# %%
class WordComplete(tf.Module, VisualWrapper):
  def __init__(self, tokenizer, transformer, max_length=512, dtype=tf.Tensor, decode_result=True):
    super().__init__()
    VisualWrapper.__init__(self, vis_on_count=None)
    self.tokenizer = tokenizer
    self.transformer = transformer
    self.max_length = max_length
    self.dtype = dtype
    self.decode_result = decode_result

  def __call__(self, input, decode=True, encoding='utf-8'):
    
    # TODO: Bug with empty strings as input
    tensor_input = tf.convert_to_tensor(input)

    if len(tensor_input.shape) == 0:
      tensor_input = tensor_input[tf.newaxis]


    tokenized_input = self.tokenizer.tokenize(tensor_input, training=False).to_tensor()

    enc_input = tokenized_input
    context = self.transformer.encode(enc_input, None, training=False)

    end = enc_input[-1][-1]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)

    for i, value in enumerate(tokenized_input[0][:-1]):
      output_array = output_array.write(i, value)
    
    out_init_len = output_array.size()

    # tf.print('real tokens: ', enc_input[0][:-1])
    # tf.print('shape of real tokens: ', tf.shape(enc_input[0][:-1]))

    for i in tf.range(out_init_len, self.max_length):
      dec_input = output_array.concat()[tf.newaxis]

      # tf.print('dec_in shape: ', tf.shape(dec_input))
      # tf.print("dec_in :", dec_input)

      decode = self.transformer.decode(context, None, dec_input, None, training=False)


      predictions = self.transformer.generator(decode, training=False)

      # tf.print('Vorhersage des Modells', tf.argmax(predictions, axis=-1))
      # tf.print('Shape of Prediction: ', tf.shape(predictions))

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.


      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.

      output_array = output_array.write(i, predicted_id[0][0])

      if predicted_id == end:
        break

    output = output_array.concat()[tf.newaxis]

    # The output shape is `(1, tokens)`.
    text = self.tokenizer.detokenize(output)  # Shape: `()`.

    tokens = self.tokenizer.lookup(output)

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    # self.transformer([encoder_input, output[:,:-1]], training=False)
    # attention_weights = self.transformer.decoder.last_attn_scores

    if self.decode_result:
      text = text.numpy()[0].decode(encoding)

    VisualWrapper.reset_counter()

    return text, tokens # , attention_weights

# %% [markdown]
# ## Text inference

# %%
inference_model = WordComplete(StoryTokenizer(reserved_tokens, vocab_path), model, max_length=32)

string = "Can you give me"

text, tokens = inference_model(string)

print(text)