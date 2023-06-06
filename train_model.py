# %% [markdown]
# # Notes
# - I finally understood, that during traingin each next token is calculated simultaneously for the whole sentence, such that no sequential processing is needed. That is of course redundant for inference. 

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
#from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

# necessary for visualization and user input
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %% [markdown]
# # Settings

# %%
# logging settings
log.basicConfig(
    format='%(asctime)s %(levelname)-8s %(processName)s %(threadName)s %(funcName)-20s %(message)s',
        # log.INFO for normal run
    # level=log.INFO,
        # log.DEBUG for diagnostics
    level=log.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# paths
bco_file_path = "datasets/bookscorpusopen/epubtxt"
vocab_path = 'datasets/vocab.txt'

# tokenizer
tokenizer_name = 'story_corpus_tokenizer'
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

# %% [markdown]
# # Utilities

# %%
def do_nothing(*args, **kwargs):
    """Placeholder for VisualWrapper"""
    pass

def clones(layer_class, N, **kwargs):
    """Produce N identical layers"""
    log.debug(f'execute with class {layer_class.__class__.__name__} and N={N}')
    return [layer_class(**kwargs) for layer_number in range(N)]

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0

# %% [markdown]
# # Visualisation

# %%
class VisualWrapper():
    """This is a mixin-Class for the tensorflow layers that enable visualization during non-training sessions."""
    should_visualize = False # globally de-/enable visualization
    instances = []          # save instances of VisualWrapper for reset_counter classmethod (see below)

    def __init__(self, vis_on_count=None, enabler=False):
        """
        Initialize a VisualWrapper instance.

        Args:
            vis_on_count (list, optional):  A list of counts on which to perform a visualizations. 
                                            If not provided, no operations will be performed on any count.
            enabler (bool, optional):       A flag used to control whether visualization is enabled. 
                                            If False, it ensures no child class does perform any visualization.
                                            Defaults to False.

        The initialized instance is appended to the `VisualWrapper.instances` list, 
        the reset_counter classmethod resets the counters of all instances in the list.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        self.counter = 0
        self.vis_on_count = vis_on_count if vis_on_count else []
        self.enabler = enabler
        VisualWrapper.instances.append(self)

    def visualize_data(self, data_x, mode, training, text=None, data_y=None, vis_diff=False):
        """
        Visualizes data_x (data_y) if tensorflow is not in training mode and global self.shoule_visualize is True.
        This only happens while self.counter is in self.vis_on_count (counter is increased by 1 with every execution)

        Args:
            data_x (unspecific often tensors):              The data that is visualized. TODO: Implement type check, to catch errors in advance
            mode (string):                                  One of the modes available in choose_func (see methods) to select the visualisation format.
            training (bool):                                Boolean parameter used by tensorflow to differentiate training and inference.
                                                            Only visualize when not in training mode.
            text (string):                                  Explanatory text giving information about the visualisation data.
                                                            Printed before visualisation is displayed.
            data_y (unspecific often tensors, optional):    TODO: Implement for multiple data visualization
            vis_diff (bool, optional):                      TODO: Implement for multiple data visualisation
        """
        log.debug(f'execute')
        if training is None:
            if self.counter in self.vis_on_count:  
                if self.should_visualize:
                    log.debug(f'visualize as training={training}, vis_counter={self.counter}, vis_conters={self.vis_on_count}, global_visualize={self.should_visualize}')
                    # if all checks for visualization are passed execute visualisation

                    # print explanatory text
                    tf.print(text)

                    # choose the correct visualization function
                    func = self.choose_func(mode)

                    # apply visualization function to data_x
                    func(data_x)

                if self.enabler:
                    # set class variable should_visualize if instance is an enabler
                    VisualWrapper.should_visualize = True
            else:
                if self.enabler:
                    # set class variable should_visualize if instance is an enabler
                    VisualWrapper.should_visualize = False
            self.counter += 1

    def choose_func(self, mode):
        """
        This function returns an executable function for the chosen 'mode'.

        Args:
            mode (string): The string indicating the visualization mode to apply.

        Returns:
            function: An executable function taking one input argument. This argument should be the data to be visualized.
        """
        log.debug(f'execute')
        if mode == 'color_bar':
            return lambda x: self.color_bar(x)
        elif mode == 'print':
            return lambda x: self.print_data(x)
        elif mode == 'reduce_dim':
            return lambda x: self.reduce_dim(x)
        else:
            # return a placeholder function, if no valid 'mode' is given.
            return do_nothing

    def color_bar(self, tensor):
        """
        Use matplotlib to plot a colorbar that visualizes the values of a 1-D-tensor.

        Args:
            tensor (tf.tensor): The tensor to be visualized
        """
        log.debug(f'execute')
        # labels for the plot TODO: Generalize such that the labels are valid for all data types.
        x_label = 'Positions'
        y_label = 'Embbeddings'

        # Assuming data[0] is a numpy array.
        # If it's a ListWrapper or another list-like object, convert it to a numpy array.
        # TODO: Doesn't work. Check for error.
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
        log.debug(f'execute')
        tf.print(data)

    def reduce_dim(self, tensor):
        """
        Reduces the dimensionality of the input tensor using PCA and plots the result.

        This function first scales the input tensor by its minimum absolute value, then applies PCA to reduce its 
        dimensionality to 3. It then creates a 3D quiver plot of the reduced data.

        Args:
            tensor (np.ndarray): The input tensor to be reduced and visualized. 

        Shows:
            A 3D matplotlib plot of the tensor after dimensionality reduction using PCA.
        """
        log.debug(f'execute')
        # Reduce the first dimension, to create a 1-D numpy array.
        array = np.squeeze(tensor, axis=0)

        # Scale the array by its minimum absolute value to normalize the data
        scaled_array = array / np.min(np.abs(array))

        # Apply PCA for dimensionality reduction.
        # This reduces the dimensions of the data to 3.
        # TODO: PCA must be trained. Alternative algorithms could be tsne or umap.
        pca = PCA(n_components=3)
        reduced_array = pca.fit_transform(scaled_array)

        # Create a new figure and a set of subplots. 
        # The figure size is set to (3,3) to maintain a square aspect ratio. 
        # TODO: Find best size for plot
        fig, ax = plt.subplots(figsize=(3, 3))
        # Add another subplot to create a 3D plot.
        ax = fig.add_subplot(111, projection='3d')

        # Create a quiver plot to visualize each point as a vector from the origin
        ax.quiver(0, 0, 0, reduced_array[:, 0], reduced_array[:, 1], reduced_array[:, 2], arrow_length_ratio=0.1)

        # Label each component (PCA dimension) on the axes.
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        # Set a title for the plot
        # TODO: Generalize the title
        ax.set_title('Embeddings')

        # Set the plot boundaries to be the maximum value in the reduced array.
        boundaries = np.max(reduced_array)
        ax.set_xlim([-boundaries, boundaries])
        ax.set_ylim([-boundaries, boundaries])
        ax.set_zlim([-boundaries, boundaries])

        # Disply the plot
        plt.show()

    @classmethod
    def reset_counter(cls):
        """Reset the counter for all instances of the class."""
        log.debug(f'execute')
        for instance in cls.instances:
            instance.counter = 0

# %% [markdown]
# # Architecture

# %% [markdown]
# ## Main Layers

# %% [markdown]
# These classes are built using the Keras Functional API, which provides more flexibility than the Sequential API for defining complex models. Each class is a subclass of tf.keras.layers.Layer, so they can be composed to build more complex layers or models. The call method of each class defines the computation that the layer performs.
# 
# These classes are designed to be components of a larger transformer model. The model itself is typically composed of an encoder and a decoder, each of which is made up of a stack of identical layers. The layers themselves contain sublayers that perform operations such as self-attention, source attention (in the case of the decoder), and position-wise feed-forward networks. These operations are encapsulated within classes like `EncoderStack`, `DecoderStack`, `EncoderLayer`, `DecoderLayer`, and `PositionwiseFeedForward`. The layer norm and dropout are applied in `ResidualSublayer` for regularizing and speeding up the training process.

# %% [markdown]
# ### Encoder Decoder Layer

# %%
class EncoderDecoder(tf.keras.Model, VisualWrapper):
    """
    Defines a Transformer model for sequence-to-sequence tasks.

    This class implements the Transformer architecture, which consists of an Encoder and Decoder, each built from multiple stacked self-attention and feedforward layers.
    Inherits from both the TensorFlow Keras Model class for building ML models and a custom VisualWrapper class for data visualization.

    Attributes:
        encoder_stack (Encoder):                The encoder component of the Transformer.
        decoder_stack (Decoder):                The decoder component of the Transformer.
        enc_embed (tf.keras.layers.Embedding):  The input embedding layer for the encoder.
        dec_embed (tf.keras.layers.Embedding):  The input embedding layer for the decoder.
        generator (tf.keras.layers.Dense):      The output linear layer.

    Note: We use two seperate embeddings, because the encoder get's the data with start token, while the decoder get's the data without start token.
    Note: To receive actual output from the model it is necessary to run call on the input and then the generator on the output.
    """

    def __init__(self, encoder_stack, decoder_stack, enc_embed, dec_embed, generator):
        """
        Initialize the EncoderDecoder model together with the parent VisualWrapper.
        The EncoderDecoder model is an visualization enabler, that means, that it enables visualization for all its sublayer, if self.counter in self.vis_on_count.

        Args:
            encoder_stack (layers.Layer):   A Encoder object, consisting of a stack of self-attention and feedforward layers.
                                            The stack size is determined within the object.
            decoder_stack (layers.Layer):   A Decoder object, consisting of a stack of self-attention, source-attention and feedforward layers.
                                            The stack size is determined within the object.
            enc_embed (layers.Layer):       An embedding layer for the encoder input.
            dec_embed (layers.Layer):       An embedding layer for the decoder input.
            generator (layers.Layer):       The final linear layer that generates predictions.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0], enabler=True)

        self.encoder_stack = encoder_stack
        self.decoder_stack = decoder_stack
        self.enc_embed = enc_embed
        self.dec_embed = dec_embed
        self.generator = generator

    def encode(self, inputs, pad_mask, training=None):
        """
        Args:
            inputs (Tensor):            Input data tensor to encode.
            pad_mask (Tensor):          Mask tensor to ignore padding tokens in the input.
            training (bool, optional):  Boolean flag indicating whether the model is in training mode. Defaults to None.

        Returns:
            Tensor:                     The output of the encoder stack.
        """
        log.debug(f'execute')
        return self.encoder_stack(self.enc_embed(inputs), 
                                  pad_mask, 
                                  training=training)

    def decode(self, enc_input, pad_mask, inputs, subseq_mask, training=None):
        """
        Args:
            enc_input (Tensor):         Encoded input data tensor to decode.
            pad_mask (Tensor):          Mask tensor to ignore padding tokens in the input.
            inputs (Tensor):            Input data tensor for the decoder.
            subseq_mask (Tensor):       Mask tensor to ignore subsequent tokens in the input.
            training (bool, optional):  Boolean flag indicating whether the model is in training mode. Defaults to None.

        Returns:
            Tensor:                     The output of the decoder stack.
        """
        log.debug(f'execute')
        return self.decoder_stack(self.dec_embed(inputs), 
                                  enc_input, 
                                  pad_mask, 
                                  subseq_mask, 
                                  training=training)

    def call(self, inputs, training=None):
        """
        Args:
            inputs (tuple):             Tuple of Tensors (enc_input (Tensor, dtype=tf.float32), 
                                                          dec_input, (Tensor, dtype=tf.float32)
                                                          pad_mask, (Tensor, dtype=tf.bool)
                                                          subseq_mask(Tensor, dtype=tf.bool)
                                                         ).
            training (bool, optional):  Boolean flag indicating whether the model is in training mode. Defaults to None.

        Returns:
            Tensor: The output of the model.
        """
        # We need to unpack the input, as tensorflows model.fit method requires the input to be passed as a single parameter,
        # but it actually contains (enc_input, dec_input, pad_mask, subseq_mask) as a tuple.
        enc_input, dec_input, pad_mask, subseq_mask = inputs

        # the following is only used to visualize model input and output data
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

# %%
class LayerNorm(layers.Layer, VisualWrapper):
    """
    Implements the Layer Normalization technique, a type of normalization performed on inputs across features.

    Inherits from both the TensorFlow Keras Layer class for building custom layers, and a custom VisualWrapper class for data visualization.
    """

    def __init__(self, input_size, eps=1e-6):
        """
        Args:
        features (int):             The size of the input data.
            eps (float, optional):  A small constant added to the variance to avoid dividing by zero. Defaults to 1e-6.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])

        # Initialize the scale and offset parameters
        self.a_2 = self.add_weight(shape=(input_size,), initializer='ones', name=self.name + "a_2")
        self.b_2 = self.add_weight(shape=(input_size,), initializer='zeros', name=self.name + "b_2")
        self.eps = eps

    def call(self, input_tensor):
        """
        Performs the layer normalization on the input data.

        Args:
            input_tensor (Tensor):  The input data.

        Returns:
            norm_out (Tensor):      The normalized data.
        """
        # Compute the mean and variance of the input data
        mean, var = tf.nn.moments(input_tensor, axes=-1, keepdims=True)

        # Compute the standard deviation
        std = tf.math.sqrt(var + self.eps)

        # Perform the layer normalization
        norm_out = self.a_2 * (input_tensor - mean) / std + self.b_2

        return norm_out

# %% [markdown]
# ### Residual Layer

# %%
class ResidualSublayer(layers.Layer, VisualWrapper):
    """
    A layer that applies a sublayer to the input, followed by dropout, and then adds the input to the result.
    This follows the 'pre-norm' variation of the Transformer architecture, where Layer Normalization is applied before the sublayer.

    !!! This layer is used to wrap the attention sublayer and the feedforward layer in the encoder stack and decoder stack. !!!
    
    Inherits from both the TensorFlow Keras Layer class for building custom layers, and a custom VisualWrapper class for data visualization.
    """

    def __init__(self, size, dropout):
        """
        Args:
            size (int):         The number of features in the input data.
            dropout (float):    The rate of dropout to apply after the sublayer.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])

        self.norm = LayerNorm(size)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_tensor, sublayer, training=None):
        """
        Applies the sublayer to the input after normalization, applies dropout, and then adds the input to the result.

        Args:
            input_tensor (Tensor):      The input data.
            sublayer (layers.Layer):    The sublayer to apply to the input data.
            training (bool):            Indicates whether the layer should behave in training mode (apply dropout) or in inference mode (do not apply dropout).

        Returns:
            residual_out (Tensor): The output data.
        """
        # Apply normalisation and sublayer
        norm_input = self.norm(input_tensor)
        sublayer_out = sublayer(norm_input)

        # If visualization is enabled for the current step, compute the sublayer output with and without dropout and visualize the difference
        # We need to check this additionally here, as we need to separately compute the dropout (dropout is only used during training not during inference),
        # but we don't want to compute dropout twice, if not necessary.
        if not training and self.counter in self.vis_on_count:
            # compute dropout even when training=False
            sublayer_dropout = self.dropout(sublayer_out, training=True)

            self.visualize_data(sublayer_dropout-sublayer_out, 
                                mode="color_bar", 
                                training=training, 
                                text="Visualize difference before/after dropout.")
            
        # compute residual output by applying dropout to the sublayer output and adding to the input
        residual_out = input_tensor + self.dropout(sublayer_out, training=training)

        return residual_out

# %% [markdown]
# ### Encoder Stack Layer

# %%
class EncoderStack(layers.Layer, VisualWrapper):
    """
    This class represents the Encoder part of the Transformer model, which is composed of a stack of identical layers.
    Each layer in the Encoder Stack consists of two sub-layers: a Multi-head Self-Attention mechanism, and a Position-wise
    Fully Connected Feed-Forward network.
    A residual connection is employed around each of the two sub-layers, followed by layer normalization.
    """

    def __init__(self, layer, N, size, **kwargs):
        """
        Inititalize the EncoderStack instance
        Args:
            layer (layers.layer):   An instance of a layer, which will be cloned N times to form the encoder stack.
            N (int):                The number of layers in the encoder stack.
            size (int):             The dimensionality of the input/ouput space of the encoder.
            **kwargs (various):     Additional keyword arguments. They contain the parameters of the layers in self.layers, such that they can
                                    be passed to the clone function that initializes the layers.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.layers = clones(layer, N, size=size, **kwargs) # Creating N identical layer stacks to form the encoder
        self.norm = LayerNorm(size)

    def call(self, input_tensor, mask, training=None):
        """
        This function propagates the input through the encoder stack, by applying succesively each layer in the self.layers attribute.
        This is aquivalent to running the attention layer and the fully connected feed-forward layer N times, 
        before finally normalising and returning an output.

        Args:
            input_tensor (Tensor): The input to the encoder.
            mask (Tensor of dtype tf.Boolean): A boolean mask tensor for padding positions within the input.
            training (bool, None): A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            (Tensor):              The output of the encoder stack.
        """
        for layer in self.layers:
            input_tensor = layer(input_tensor, mask, training=training)

        encoder_out = self.norm(input_tensor, training=training)

        return encoder_out

# %% [markdown]
# ### Encoder Layer

# %%
class EncoderLayer(layers.Layer, VisualWrapper):
    """
    This class represents a single layer within the Encoder stack of the Transformer model.
    Each EncoderLayer consists of two sub-layers: 
        - a Multi-head Self-Attention mechanism, 
        - a Position-wise Fully Connected Feed-Forward network.
    A residual connection is employed around each of the two sub-layers.
    
    Note:   The residual sublayers do themselves not contain sublayers, because of two reasons:
                - Like that we can clone the ResidualSublayer, instead of having to write out each single sublayer
                - During forward pass, we need to pass different information to the sublayers e.g. mask, training, x, context.
                  This process is simplified if the ResidualSublayer can be skipped.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Initializes the EncoderLayer
        Args:
            size (int):                    The dimensionality of the input/output space of the encoder.
            self_attn (layers.Layer):      The Multi-head Self-Attention mechanism.
            feed_forward (layers.Layer):   The Position-wise Fully Connected Feed-Forward network.
            dropout (float):               The dropout rate to be applied to the output during training.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSublayer, 
                               N=2, 
                               size=size, 
                               dropout=dropout)

    def call(self, input_tensor, mask, training=None):
        """
        This function propagates the input through the attention layer and the feed-forward layer.

        The Self-Attention mechanism (sublayer[0]) takes three times the input_tensor as input for query, key, value
        it hiddes the padding through a padding mask, passed through the mask argument, and returns the rearranged output.

        The Feed-Forward network takes the output of the Self-Attention mechanism and creates meaningful information for the next Encoder-Layer.
        
        Args:
            input_tensor (Tensor):              The input to the encoder layer.
            mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding positions within the input.
            training (bool, optional):          A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            ff_output (Tensor):                 The output of the encoder layer.
        """
        attn_output = self.sublayer[0](input_tensor, 
                                        lambda x: self.self_attn(x, x, x, 
                                                                 mask, 
                                                                 training=training), 
                                        training=training)
        ff_output = self.sublayer[1](attn_output, 
                                     lambda x: self.feed_forward(x, 
                                                                 training=training), 
                                     training=training)
        return ff_output

# %% [markdown]
# ### Decoder Stack Layer

# %%
class DecoderStack(layers.Layer, VisualWrapper):
    """
    This class represents the Decoder part of the Transformer model, which is composed of a stack of identical layers.
    Each layer in the Decoder Stack consists of three sub-layers: 
        - a Masked Multi-head Self-Attention mechanism, 
        - a Multi-head Self-Attention mechanism over the Encoder's output,
        - a Position-wise Fully Connected Feed-Forward network.
    A residual connection is employed around each of the three sub-layers, followed by layer normalization.
    """

    def __init__(self, layer, N, size, **kwargs):
        """
        Inititalize the DecoderStack instance
        Args:
            layer (layers.layer):   An instance of a layer, which will be cloned N times to form the decoder stack.
            N (int):                The number of layers in the decoder stack.
            size (int):             The dimensionality of the input/output space of the decoder.
            **kwargs (various):     Additional keyword arguments. They contain the parameters of the layers in self.layers, such that they can
                                    be passed to the clone function that initializes the layers.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.layers = clones(layer, N, size=size, **kwargs)
        self.norm = LayerNorm(size)

    def call(self, input_tensor, enc_memory_tensor, src_mask, tgt_mask, training=None):
        """
        This function propagates the input through the decoder stack, by applying succesively each layer in the self.layers attribute.
        This is equivalent to running the masked attention layer, the attention layer over the encoder's output, 
        and the fully connected feed-forward layer N times, before finally normalising and returning an output.

        Args:
            input_tensor (Tensor):                  The input to the decoder.
            enc_memory_tensor (Tensor):             The output of the encoder, serves as the "memory" in the Transformer model.
            src_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding positions within the source input.
            tgt_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding and preventing "future" information 
                                                    in attenting to the source input.
            training (bool, None):                  A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            decoder_out (Tensor):                   The output of the decoder stack.
        """
        for layer in self.layers:
            input_tensor = layer(input_tensor, 
                                 enc_memory_tensor, 
                                 src_mask, tgt_mask, 
                                 training=training)
            
        decoder_out = self.norm(input_tensor, training=training)
        
        return decoder_out

# %% [markdown]
# ### Decoder Layer

# %%
class DecoderLayer(layers.Layer, VisualWrapper):
    """
    This class represents a single layer within the Decoder stack of the Transformer model.
    Each DecoderLayer consists of three sub-layers:
        - a Masked Multi-head Self-Attention mechanism,
        - a Multi-head Self-Attention mechanism that interacts with the output of the encoder,
        - a Position-wise Fully Connected Feed-Forward network.
    A residual connection is employed around each of the three sub-layers.
    
    Note: The residual sublayers do not themselves contain sublayers, because of two reasons:
      - This way, we can clone the ResidualSublayer, instead of having to write out each single sublayer.
      - During the forward pass, we need to pass different information to the sublayers e.g. masks, training, context. 
        This process is simplified if the ResidualSublayer can be skipped.
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Initializes the DecoderLayer
        Args:
            size (int):                    The dimensionality of the input/output space of the decoder.
            self_attn (layers.Layer):      The Masked Multi-head Self-Attention mechanism.
            src_attn (layers.Layer):       The Masked Multi-head Source-Attention mechanism that interacts with the encoder output.
            feed_forward (layers.Layer):   The Position-wise Fully Connected Feed-Forward network.
            dropout (float):               The dropout rate to be applied to the output during training.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSublayer, N=3, size=size, dropout=dropout)

    def call(self, input_tensor, enc_memory, src_mask, tgt_mask, training=None):
        """
        This function propagates the input through the decoder layer.

        The Masked Self-Attention mechanism (sublayer[0]) takes three times the input_tensor as input for query, key, value.
        It hides the padding and future positions from affecting the current token's attention calculation through a combined padding and lookahead mask, 
        passed through the tgt_mask argument, and returns the rearranged output.

        The Encoder-Decoder Attention mechanism (sublayer[1]) takes the output from the previous Masked Self-Attention mechanism as the query and the encoder 
        output (memory) as both the key and the value. It also employs a padding mask (src_mask) on the encoder output and returns the attention-combined output.

        The Feed-Forward network (sublayer[2]) takes the output of the Encoder-Decoder Attention mechanism and creates meaningful information for the next Decoder-Layer.
        
        Args:
            input_tensor (Tensor):                             The input to the decoder layer.
            enc_memory (Tensor):                        The output of the encoder, serves as the "memory" in the Transformer model.
            src_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding positions within the source input.
            tgt_mask (Tensor of dtype tf.Boolean):  A boolean mask tensor for padding and preventing "future" information in self-attention mechanism within the target input.
            training (bool, optional):              A boolean indicating whether to run the layer in training mode or inference mode.

        Returns:
            ff_out (Tensor):                The output of the decoder layer.
        """
        self_attn_out = self.sublayer[0](input_tensor, 
                                         lambda x: self.self_attn(x, x, x, 
                                                                  tgt_mask, 
                                                                  training=training),
                                         training=training)
        src_attn_out = self.sublayer[1](self_attn_out, 
                                        lambda x: self.src_attn(x, enc_memory, enc_memory, 
                                                                src_mask,
                                                                training=training),
                                        training=training)
        ff_out = self.sublayer[2](src_attn_out, 
                                  lambda x: self.feed_forward(x,
                                                              training=training),
                                  training=training)

        return ff_out

# %% [markdown]
# ## Sublayers

# %% [markdown]
# ### Feedforward Layer

# %%
class PositionwiseFeedForward(layers.Layer, VisualWrapper):
    """
    Implements the Position-wise Feed-Forward Network (FFN) for the Transformer model.
    The FFN consists of two fully connected layers with a ReLU activation in between.

    Attributes:
        dense_in (Dense):    First dense layer.
        dense_out (Dense):   Second dense layer.
        dropout (Dropout):   Dropout layer.
    """


    def __init__(self, d_model, d_ff, dropout=0.1):
        """
            Args:
                d_model (int):      Output dimensionality of the Transformer.
                d_ff (int):         Inner-layer dimensionality.
                dropout (float):    Dropout rate after the ReLU activation.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.dense_in = layers.Dense(d_ff)
        self.dense_out = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout)

    def call(self, input_tensor, training=None):
        """
        Forward pass for the FFN. Applies both layers and ReLU activation to the input tensor.
        Dropout inbetween the layers, as the ResidualLayer that wraps around will again perform a dropout

        Args:
            x (Tensor): Input tensor.
            training (bool, optional): Indicates whether to run the layer in training mode or inference mode.

        Returns:
            (Tensor): Output tensor.
        """
        return self.dense_out(self.dropout(tf.nn.relu(self.dense_in(input_tensor)), training=training))

# %% [markdown]
# ### Generator Layer

# %%
class Generator(layers.Layer, VisualWrapper):
    """
    This class serves as the final layer of the Transformer model, generating the predicted output.
    It applies a dense layer to the final output of the Transformer model and then a log softmax function 
    across the vocabulary dimension. This results in a distribution over the possible output tokens for each 
    position in the sequence, where the value of each token is the log probability of that token being the 
    output for that position.

    Attributes:
        proj (Dense): Dense layer that is applied to the final output of the Transformer model. It increases 
        the dimensionality of the input to be the size of the vocabulary.
    """

    def __init__(self, vocab):
        """
        Args:
            vocab (int): Size of the output vocabulary.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])

        self.proj = layers.Dense(vocab)

    def call(self, input_tensor, training=None):
        """
        This method applies the Dense layer and log softmax function to its input.
        
        Args:
            input_tensor (Tensor):      The input tensor, which is the final output from the Transformer model.
            training (bool, optional):  Indicates whether to run the layer in training mode or inference mode.

        Returns:
            result (Tensor):    A tensor of the same shape as the input, but the last dimension is now the size 
                                of the vocabulary. Each value in this tensor is the log probability of the corresponding token 
                                being the output for the position in the sequence.
        """
        result = tf.nn.log_softmax(self.proj(input_tensor), axis=-1)

        if not training:
            self.visualize_data(result, 
                                'color_bar', 
                                text=f"This is the data from {self.__class__.__name__}", 
                                training=training)
        return result

# %% [markdown]
# ### Attention Layer
# 
# - If you try to understand this code, start with the MultiHeadAttention class.

# %%
def attention(query, key, value, mask=None, dropout=None, training=None):
    """
    Compute 'Scaled Dot Product Attention'

    The attention function computes a weighted sum of the value vectors, with the weights being determined by the similarity of the
    query vector with the corresponding key vector. The dot product of the query and key serves as the measure of similarity, and is scaled
    by the square root of the dimension of the key vector to avoid the issue of the dot product growing large for large dimensions.

    Args:
        query, key, value (Tensor):                     The query, key and value vectors. 
                                                        These typically have shape (batch_size, num_heads, seq_len, depth).
                                                        (seq_len as we want to calculate the attention for each position simultaneously)
        mask (Tensor of dtype tf.Boolean, optional):    A mask to apply to the attention scores before softmax, 
                                                        in order to prevent attention to certain positions. 
                                                        The shape should be broadcastable to shape (batch_size, num_heads, seq_len, seq_len???).
        dropout (Dropout, optional):                    Dropout layer to apply to the attention scores after softmax.
        training (bool, optional):                      Whether the model is in training mode.

    Returns:
        output (Tensor):                                The result of applying attention mechanism to the value vectors.
        p_attn (Tensor):                                The attention weights after applying softmax and dropout.
    """
    log.debug(f'execute')
    # Compute the dot product of the query and key vectors and scale by sqrt(d_k)
    d_k = tf.cast(query.shape[-1], dtype=tf.float32)
    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / tf.sqrt(d_k)

    # Apply the mask to the scores before softmax
    if mask is not None:
        mask = tf.cast(mask, dtype=tf.bool)
        scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

    # Apply softmax to the scores to get the attention weights
    p_attn = tf.nn.softmax(scores, axis=-1)

    # Apply dropout to the attention weights
    if dropout is not None:
        p_attn = dropout(p_attn, training=training)

    # Compute the weighted sum of the value vectors, using the attention weights
    attn_out = tf.matmul(p_attn, value)

    return attn_out, p_attn

# %%
class MultiHeadedAttention(layers.Layer, VisualWrapper):
    """
    MultiHeadedAttention layer is a key part of the Transformer model, enabling it to pay attention to different parts of the input for each output.
    This is achieved by having multiple 'heads', each of which independently computes a scaled dot product attention over the input.
    The outputs of these heads are then concatenated and linearly transformed to produce the final output.

    Attributes:
        d_k (int):                          Dimensionality of the query, key, and value vectors, 
                                            which should be identical for each head.
        h (int):                            Number of heads.
        query, key, value, linear (Dense):  These are the layers that perform the linear transformations for the input.
        attn (Tensor, optional):            Tensor storing the attention values from the last forward pass.
        dropout (Dropout):                  Dropout layer applied after the attention.
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h (int):                    Number of attention heads.
            d_model (int):              Dimensionality of the model.
            dropout (float, optional):  Dropout rate.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0, 1])

        assert d_model % h == 0 # make sure the number of attention heads are such that they can equally distribute over the input tensor
        self.d_k = d_model // h # determine the size for the attention heads
        self.h = h
        self.query, self.key, self.value, self.linear = clones(layers.Dense, N=4, units=d_model)
        self.attn = None
        self.dropout = layers.Dropout(dropout)

    def call(self, query, key, value, mask=None, training=None):
        """
        Forward pass for the MultiHeadedAttention layer.
        Applies linear transformations to the input, applies scaled dot product attention, then applies dropout, concatenates the heads,
        and applies a final linear transformation.

        Args:
            query, key, value (Tensor):                     Input tensors. Value and query are (in our case) always the same.
            mask (Tensor of dtype tf.Boolean, optional):    Boolean mask tensor for padding positions within the input.
            training (bool, optional):                      Indicates whether to run the layer in training mode or inference mode.

        Returns:
            result (Tensor):                                The output tensor.
        """

        if mask is not None:
            # Same mask applied to all h heads
            mask = tf.expand_dims(mask, 1)

        # find out how many batches are processed in parallel
        nbatches = tf.shape(query)[0]

        # Transform the input data into a shape that can be used as matrix input for the attention function.
        # The original size is d_model, the trainable matrices self.query, self.key, self.value transform this input tensor
        # into a tensor of the same size, but now we have to think of it as being of size h * d_k. Such that each section of size d_k,
        # will be passed through the attention mechanism independently. That is why all this transformations have to be done afterwards.
        # [nbatches, -1, self.h, self.d_k] does split the tensor into h smaller tensors of size d_k 
        # (nbatches is only there for working with batches of data). The Permutation ensures, that the h tensors are of shape (1, d_k), 
        # such that they can be processed.
        query, key, value = [
            tf.transpose(tf.reshape(lin_layer(input), [nbatches, -1 , self.h, self.d_k]), perm=[0, 2, 1, 3]) 
            for lin_layer, input in zip([self.query, self.key, self.value], (query, key, value))
        ]

        att_out, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, training=training)

        # Now we reverse the whole process and reshape the output into vectors of shape (nbatches, 1, d_model) again.
        att_out = tf.reshape(tf.transpose(att_out, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))

        # visualization functions
        if not training:
            self.visualize_data(att_out, 
                                mode="color_bar",
                                training=training,
                                text=f"This is the output of {self.__class__.__name__}.")
            self.visualize_data(self.attn,
                                mode="color_bar",
                                training=training,
                                text=f"This is the attention applied by {self.__class__.__name__}")

        # This finally mixes the results of the different heads together into one output vector
        linear_output = self.linear(att_out)

        return linear_output

# %% [markdown]
# ### Positional Embedding Layer

# %%
def positional_encoding(length, depth):
    """
    Generate positional encoding for a given length and depth to provide positional information.
    Positional encoding is a technique where each position in the input sequence is assigned a 
    unique vector representation.

    The encoding vector alternates between the sine and cosine functions of different 
    frequencies, which allows the model to distinguish the position of the inputs.

    The positional encoding function uses a specific ratio to scale down the angle rates 
    exponentially (1 / (10000**(depth/depth))). It means that for lower dimensions in the 
    positional encoding, the angle rate is high which means the positional encoding is 
    changing rapidly for lower dimensions. For higher dimensions, the angle rate is low 
    which means the positional encoding is changing slowly. It gives a balance between 
    low and high frequency information.

    Args:
        length (int):   Length of the sequence for which positional encoding is to be generated.
        depth (int):    The number of dimensions for the positional encoding. Equals the embedding size.

    Returns:
        Tensor:         A 2D Tensor of shape (length, depth) containing the positional encoding vectors.
    """
    log.debug(f'execute')
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]    # Creates a numpy array of shape (sequence_length, 1)
                                                    # filled with the numbers 1 to sequence length
    depths = np.arange(depth)[np.newaxis, :]/depth  # Creates a numpy array of shape (1, depth/2)
                                                    # filled with the numbers 1 to depth/2 divided by depth

    angle_rates = 1 / (10000**depths) 
    angle_rads  = positions * angle_rates           # broadcasting such that now element [i,j] is pos(i) * angle(j)

    # as we have above chosen depth/2 we can now concatenate sines and cosines to aquire an vectore of size depth
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

# %%
class PositionalEmbedding(layers.Layer, VisualWrapper):
    """
    A Keras layer to apply positional encoding on top of embeddings.

    This layer creates embeddings for discret input vectors created by a tokenizer
    and applies positional encoding to these embeddings to provide positional information.
    The positional encoding is pre-computed in the constructor for efficiency and it is added to the output 
    of the embedding layer in the `call` method. The dropout is used to train the embeddings.
    """
    def __init__(self, vocab_size, d_model, dropout):
        """
        Initializes Positional Embeddings

        Args:
            vocab_size (int):   The size of the input token vector.
            d_model (int):      The dimension used for the embeddings and positional encoding passed to the model.
            dropout (float):    Value used for dropout.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0,1,2])
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.dropout = layers.Dropout(dropout)

        # calculate positional encoding
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, input_token_vec, training=None):
        """
        Performs the forward pass for the embedding and positional encoding layers.
        
        Args:
            input_token_vec (Tensor):   Input tensor of shape `(batch_size, sequence_length)`.
            training (bool, optional):  Indicator for the mode (training or inference) of the model.

        Returns:
            y (Tensor):     The output tensor after applying embedding, positional encoding, and dropout. 
                            It has the shape of `(batch_size, sequence_length, d_model)`.
        """

        length = tf.shape(input_token_vec)[1]

        x_emb = self.embedding(input_token_vec) # is now a tensor of shape (batch_size, length, d_model)
        x_emb_scale = x_emb * tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # This factor sets the relative scale of the embedding and positional_encoding
        
        y = self.dropout(x_emb_scale + self.pos_encoding[tf.newaxis, :length, :])

        if not training:
            self.visualize_data(x_emb, 
                                mode='color_bar', 
                                text=f"This is the embedding of the input to {self.__class__.__name__}.", 
                                training=training)
            self.visualize_data(y, 
                                mode='color_bar', 
                                text=f"This is the embedding of the input to {self.__class__.__name__} with added positional encoding.", 
                                training=training)
            self.visualize_data(x_emb-y, 
                                mode='color_bar', 
                                text=f"Here you can see the difference between both.", 
                                training=training)
        return y

# %% [markdown]
# # Training Setup

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Tokenizer

# %%
class StoryTokenizer(tf.Module, VisualWrapper):
    """
    The StoryTokenizer class is designed to perform tokenization and detokenization tasks using the BERT tokenizer.
    
    Methods:
        tokenize:               Tokenize a string with BERT Tokenizer, add [Start] and [End] tokens.
        detokenize:             Detokenize a token vector, clean the string of the reserved tokens.
        lookup:                 Return the tokens a string is composed of.
        add_start_end:          Add [Start], [End] toknes to a raggend token vector.
        cleanup_text:           Remove reserved tokens from a string.
        get_vocab_size:         Return the length of the vocabulary used by the tokenizer.
        get_vocab_path:         Return the path of the vocabulary filee.
        get_reserved_tokens:    Return a list of all reserved tokens.
    """
    def __init__(self, reserved_tokens, vocab_path):    
        """
        Initialize a StoryTokenizer

        Args:
            reserved_tokens (list of strings):  A list of strings with special tokens
            vocab_path (string):                The path to the vocabulary file
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])

        self.tokenizer = tf_text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        # read in the vocabulary from file.
        vocab = pathlib.Path(vocab_path).read_text(encoding='utf-8').splitlines()
        self.vocab = tf.Variable(vocab)        

    def tokenize(self, strings, training=None):
        """
        Tokenizes the input strings and adds start and end tokens.

        Args:
            strings (tf.Tensor):        The strings to be tokenized.
            training (bool, optional):  If True, the model is in training mode. Defaults to None.

        Returns:
            out (tf.RaggedTensor):      The tokenized strings with added start and end tokens.
        """
        log.debug(f'execute')
        encoded = self.tokenizer.tokenize(strings)
        merged_enc = encoded.merge_dims(-2, -1)
        out = self.add_start_end(merged_enc)

        if not training:
            self.visualize_data(self.lookup(out),
                                mode='print', 
                                text=f"This is the data from {self.__class__.__name__}", 
                                training=training)

        return out
    
    def detokenize(self, tokenized, training=None):
        """
        Detokenizes the input token IDs back into text strings.
        Any reserved tokens (except for "[UNK]") are removed from the detokenized text.

        Args:
            tokenized (tf.RaggedTensor): The token IDs to be detokenized.
            training (bool, optional): If True, the model is in training mode. Defaults to None.

        Returns:
            tf.Tensor: The detokenized text.
        """
        log.debug(f'execute')
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(self._reserved_tokens, words)
    
    def lookup(self, token_ids):
        """
        Converts token IDs to their corresponding token strings from the vocabulary.

        Args:
            token_ids (tf.RaggedTensor or tf.Tensor): The token IDs to be converted.

        Returns:
            tf.RaggedTensor or tf.Tensor: The corresponding token strings.
        """
        log.debug(f'execute')
        return tf.gather(self.vocab, token_ids)

    @staticmethod
    def add_start_end(ragged):
        """
        Adds start and end tokens to the input token IDs.

        Args:
            ragged (tf.RaggedTensor): The input token IDs.

        Returns:
            tf.RaggedTensor: The token IDs with added start and end tokens.
        """
        log.debug(f'execute')
        # Create vectores for the [Start] and [End] tokens.
        START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
        END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

        # fill up dim 0 and concat in dim 1 to handle batches.
        count = ragged.bounding_shape()[0]
        starts = tf.fill([count, 1], START)
        ends = tf.fill([count, 1], END)
        return tf.concat([starts, ragged, ends], axis=1)

    @staticmethod
    def cleanup_text(reserved_tokens, token_txt):
        """
        Removes any reserved tokens (except for "[UNK]") from the input text.

        Args:
            reserved_tokens (list of str): The list of reserved tokens.
            token_txt (tf.Tensor): The input text.

        Returns:
            tf.Tensor: The cleaned up text.
        """
        log.debug(f'execute')
        # Create a regular expression searching for reserved tokens
        bad_tokens = list(filter(lambda token: token != "[UNK]", reserved_tokens))
        bad_tokens_re = "|".join(bad_tokens)

        # Search and delete reserved tokens from the token_txt tensor
        bad_cells = tf.strings.regex_full_match(token_txt, bad_tokens_re)
        ragged_result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # join the text
        result = tf.strings.reduce_join(ragged_result, separator=' ', axis=-1)

        return result
    
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]
    
    def get_vocab_path(self):
        return self._vocab_path
    
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

# %% [markdown]
# ### Dataset generator

# %%
class DatasetGenerator():
    """
    A class to generate TensorFlow datasets for Transformer models from text files.
    The txt_files_to_lines_gen generates lines, that are fitted into a certain length by
    lines_to_fit_sentences (it combines follow-up sentences, if they don't exceed the limit together)..
    generate_datasets and prepare_datapoint are used to generate the kind of data necessary for our model:
    (src, tgt, src_mask, tgt_mask), label. Here src, tgt and label are similar tensors, but shifted right or left
    and with or without [Start] or [End] tokens. The src_mask is a padding mask and the tgt_mask is a subsequent mask.
    """

    def __init__(self,
                 tokenizer, 
                 buffer_size=20000, 
                 batch_size=64,
                 train_val_test_size = (0, 0, 0),
                 max_padding=512, 
                 pad_id=0):
        """
        Constructor for the DatasetGenerator class.
        
        Args:
            tokenizer:          Instance of the tokenizer to be used.
            buffer_size (int):  Number of elements from the dataset from which the new dataset will sample.
                                This is crucial for randomisation, as the dataset is not shuffled further than buffer_size allows.
            batch_size (int):   Number of elements per batch in the dataset.
            max_padding (int):  The maximum sequence length, shorter sequences will be padded with pad_id.
            pad_id (int):       ID to be used for padding shorter sequences.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_size, self.val_size, self.test_size = train_val_test_size
        self.max_padding = max_padding
        self.pad_id = pad_id
        self.dataset = None

    def txt_files_to_lines_gen(self, file_path):
        """
        Generator function that yields lines from text files in a directory.

        Args:
            file_path (str):    Path to the directory containing the text files.

        Yields:
            str:                A line from a text file.
        """
        log.debug(f'execute')
        path = pathlib.Path(file_path)

        for file in path.iterdir():
            if file.is_file():
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        yield line.strip()

    def lines_to_fit_sentences(self, sentences, length):
        """
        Generator function that combines sentences so that the combined sentence is close to a certain length.
        
        Args:
            sentences (iterator):   An iterator that yields sentences.
            length (int):           The maximum length for combined sentences.
            
        Yields:
            str:                    A combined sentence.
        """
        log.debug(f'execute')
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
    
    def generate_dataset(self, file_path, train_val_test_size = None):
        """
        Generates a tokenized, batched TensorFlow dataset from text files.

        Args:
            file_path (str): Path to the directory containing the text files.

        Returns:
            tf.data.Dataset: The generated dataset. It contains the following data: (src, tgt, src_mask, tgt_mask), label
        """
        log.debug(f'execute')

        if train_val_test_size is not None:
            self.train_size, self.val_size, self.test_size = train_val_test_size

        # Create a Dataset from the text file
        lines_gen = self.txt_files_to_lines_gen(file_path)
        fit_sentence_gen = self.lines_to_fit_sentences(lines_gen, self.max_padding)
        log.debug(f'generators set up')
        
        dataset = tf.data.Dataset.from_generator(lambda: fit_sentence_gen, 
                                                 output_signature=tf.TensorSpec(shape=(), 
                                                                                dtype=tf.string))
        log.debug(f'dataset created')

        # Tokenize the whole dataset with the pre-trained tokenizer and apply our data preparation method.
        train_data = (dataset
                        .skip(0)
                        .take(self.train_size)
                        .repeat()
                        .shuffle(self.buffer_size)
                        .batch(self.batch_size)
                        .map(lambda x: self.prepare_datapoint(x), 
                             num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        val_data = (dataset
                    .skip(self.train_size)
                    .take(self.val_size)
                    .repeat()
                    .shuffle(self.buffer_size)
                    .batch(self.batch_size)
                    .map(lambda x: self.prepare_datapoint(x), 
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        test_data = (dataset
                    .skip((self.train_size + self.val_size))
                    .take(self.test_size)
                    .repeat()
                    .shuffle(self.buffer_size)
                    .batch(self.batch_size)
                    .map(lambda x: self.prepare_datapoint(x), 
                         num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(buffer_size=tf.data.AUTOTUNE))

        self.dataset = train_data, val_data, test_data
        log.debug(f'dataset processed')
        
        return self.dataset
    
    def multi_device_generate_dataset(self, input_context, file_path, train_val_test_size = None):
        """
        Generates a tokenized, batched TensorFlow dataset from text files.

        Args:
            file_path (str): Path to the directory containing the text files.

        Returns:
            tf.data.Dataset: The generated dataset. It contains the following data: (src, tgt, src_mask, tgt_mask), label
        """
        log.debug(f'execute')

        per_replica_batch_size = input_context.get_per_replica_batch_size(self.batch_size)

        if train_val_test_size is not None:
            self.train_size, self.val_size, self.test_size = train_val_test_size

        # Create a Dataset from the text file
        lines_gen = self.txt_files_to_lines_gen(file_path)
        fit_sentence_gen = self.lines_to_fit_sentences(lines_gen, self.max_padding)
        log.debug(f'generators set up')
        
        dataset = tf.data.Dataset.from_generator(lambda: fit_sentence_gen, 
                                                 output_signature=tf.TensorSpec(shape=(), 
                                                                                dtype=tf.string))
        dataset.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
        log.debug(f'dataset created')

        # Tokenize the whole dataset with the pre-trained tokenizer and apply our data preparation method.
        train_data = (dataset
                        .repeat()
                        .shuffle(self.buffer_size)
                        .batch(per_replica_batch_size)
                        .map(lambda x: self.prepare_datapoint(x), 
                             num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(buffer_size=tf.data.AUTOTUNE))

        self.dataset = train_data
        log.debug(f'dataset processed')
        
        return self.dataset
    def prepare_datapoint(self, batch):
        """
        Prepares a datapoint for the transformer model by tokenizing and creating the necessary masks.

        Args:
            data_point (str): A sentence or text to be prepared.

        Returns:
            tuple: A tuple containing source tokens, target tokens and their respective masks, and label tokens.
        """
        log.debug(f'execute')
        src_tokens = self.tokenizer.tokenize(batch)
        # Shorten tgt and label in order to remove [Start], [End] tokens
        tgt_tokens = src_tokens[:, :-1]
        label_tokens = src_tokens[:, 1:]
        
        # Fill the data to same size tensors.
        src = src_tokens.to_tensor(shape=[None, self.max_padding], 
                                   default_value=self.pad_id)
        tgt = tgt_tokens.to_tensor(shape=[None, self.max_padding], 
                                   default_value=self.pad_id)
        label = label_tokens.to_tensor(shape=[None, self.max_padding], 
                                       default_value=self.pad_id)
        
        # padding mask for source and subsequent mask for tgt
        # the masks are to be passed with the data through the model instead of reacreating it every time the model runs.
        src_mask = (src != self.pad_id)[:, np.newaxis, :]
        tgt_mask = self.make_subseq_mask(tgt)

        return (src, tgt, src_mask, tgt_mask), label
  
    def make_subseq_mask(self, tgt):
        """
        Creates a mask for the transformer model to avoid using future tokens and padding.

        Args:
            tgt (tf.Tensor): Tensor of target tokens.

        Returns:
            tf.Tensor: The mask tensor.
        """
        log.debug(f'execute')
        tgt_mask = (tgt != self.pad_id)[:, np.newaxis, :]
        tgt_mask = tf.logical_and(tgt_mask, subsequent_mask(tgt.shape[-1]))
        return tgt_mask

# %%
class DatasetGeneratorAlt():
    """
    A class to generate TensorFlow datasets for Transformer models from text files.
    The txt_files_to_lines_gen generates lines, that are fitted into a certain length by
    lines_to_fit_sentences (it combines follow-up sentences, if they don't exceed the limit together)..
    generate_datasets and prepare_datapoint are used to generate the kind of data necessary for our model:
    (src, tgt, src_mask, tgt_mask), label. Here src, tgt and label are similar tensors, but shifted right or left
    and with or without [Start] or [End] tokens. The src_mask is a padding mask and the tgt_mask is a subsequent mask.
    """

    def __init__(self,
                 tokenizer, 
                 buffer_size=20000, 
                 batch_size=64,
                 train_val_test_size = (0, 0, 0),
                 max_padding=128, 
                 pad_id=0):
        """
        Constructor for the DatasetGenerator class.
        
        Args:
            tokenizer:          Instance of the tokenizer to be used.
            buffer_size (int):  Number of elements from the dataset from which the new dataset will sample.
                                This is crucial for randomisation, as the dataset is not shuffled further than buffer_size allows.
            batch_size (int):   Number of elements per batch in the dataset.
            max_padding (int):  The maximum sequence length, shorter sequences will be padded with pad_id.
            pad_id (int):       ID to be used for padding shorter sequences.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_size, self.val_size, self.test_size = tuple(map(lambda x: x * self.batch_size, train_val_test_size))
        self.max_padding = max_padding
        self.pad_id = pad_id
        self.dataset = None

    def txt_files_to_lines_gen(self, file):
        """
        Generator function that yields lines from text files in a directory.

        Args:
            file_path (str):    Path to the directory containing the text files.

        Yields:
            str:                A line from a text file.
        """
        log.debug(f'execute')
        with open(file, 'r') as f:
            for line in f:
                yield line.strip()

    def generate_dataset(self, file_path, train_val_test_size = None):
        """
        Generates a tokenized, batched TensorFlow dataset from text files.

        Args:
            file_path (str): Path to the directory containing the text files.

        Returns:
            tf.data.Dataset: The generated dataset. It contains the following data: (src, tgt, src_mask, tgt_mask), label
        """
        log.debug(f'execute')
        if train_val_test_size is not None:
            self.train_size, self.val_size, self.test_size = tuple(map(lambda x: x * self.batch_size, train_val_test_size))

        log.debug(f'generators set up')

        path = pathlib.Path(file_path)
        file_list = [str(file) for file in path.iterdir() if file.is_file()]

        files = tf.data.Dataset.from_tensor_slices(file_list)

        # TODO: Set up a cycle length equal to the number of GPU devices.
        dataset = files.interleave(lambda x: tf.data.Dataset.from_generator(
                self.txt_files_to_lines_gen,
                args=(x,),
                output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
                ),
            num_parallel_calls = tf.data.AUTOTUNE,
            deterministic=False
        )
        log.debug(f'dataset created')
        
        # Tokenize the whole dataset with the pre-trained tokenizer and apply our data preparation method.
        train_data = (dataset
                        .skip(0)
                        .take(self.train_size)
                        .repeat()
                        .shuffle(self.buffer_size)
                        .batch(self.batch_size)
                        .map(lambda x: self.prepare_datapoint(x), 
                             num_parallel_calls=tf.data.AUTOTUNE,
                             deterministic=False)
                        .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        val_data = (dataset
                    .skip(self.train_size)
                    .take(self.val_size)
                    .repeat()
                    .shuffle(self.buffer_size)
                    .batch(self.batch_size)
                    .map(lambda x: self.prepare_datapoint(x), 
                         num_parallel_calls=tf.data.AUTOTUNE,
                         deterministic=False)
                    .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        test_data = (dataset
                    .skip(self.train_size + self.val_size)
                    .take(self.test_size)
                    .repeat()
                    .shuffle(self.buffer_size)
                    .batch(self.batch_size)
                    .map(lambda x: self.prepare_datapoint(x), 
                         num_parallel_calls=tf.data.AUTOTUNE,
                         deterministic=False)
                    .prefetch(buffer_size=tf.data.AUTOTUNE))

        self.dataset = train_data, val_data, test_data
        log.debug(f'dataset processed')
        
        return self.dataset
    
    def prepare_datapoint(self, data_point):
        """
        Prepares a datapoint for the transformer model by tokenizing and creating the necessary masks.

        Args:
            data_point (str): A sentence or text to be prepared.

        Returns:
            tuple: A tuple containing source tokens, target tokens and their respective masks, and label tokens.
        """
        log.debug(f'execute')
        src_tokens = self.tokenizer.tokenize(data_point)
        # Shorten tgt and label in order to remove [Start], [End] tokens
        tgt_tokens = src_tokens[:, :-1]
        label_tokens = src_tokens[:, 1:]
        
        # Fill the data to same size tensors.
        src = src_tokens.to_tensor(shape=[None, self.max_padding], 
                                   default_value=self.pad_id)
        tgt = tgt_tokens.to_tensor(shape=[None, self.max_padding], 
                                   default_value=self.pad_id)
        label = label_tokens.to_tensor(shape=[None, self.max_padding], 
                                       default_value=self.pad_id)
        
        # padding mask for source and subsequent mask for tgt
        # the masks are to be passed with the data through the model instead of reacreating it every time the model runs.
        src_mask = (src != self.pad_id)[:, np.newaxis, :]
        tgt_mask = self.make_subseq_mask(tgt)

        return (src, tgt, src_mask, tgt_mask), label
  
    def make_subseq_mask(self, tgt):
        """
        Creates a mask for the transformer model to avoid using future tokens and padding.

        Args:
            tgt (tf.Tensor): Tensor of target tokens.

        Returns:
            tf.Tensor: The mask tensor.
        """
        log.debug(f'execute')
        tgt_mask = (tgt != self.pad_id)[:, np.newaxis, :]
        tgt_mask = tf.logical_and(tgt_mask, subsequent_mask(tgt.shape[-1]))
        return tgt_mask

# %% [markdown]
# ## Train functions

# %% [markdown]
# ### Loss function

# %%
class LabelSmoothingLoss(layers.Layer, VisualWrapper):
    """
    This class represents a loss function layer that applies label smoothing to prevent overconfidence 
    in the model's predictions. This is done by replacing the 0s and 1s in the labels with smoothed values, 
    such that the model learns to be less confident and thus, more robust.

    Methods:
        call(x, target): Calculates and returns the loss given the model's output `x` and the target labels.

    Example:
        >>> loss_func = LabelSmoothingLoss(vocab_size=5000, padding_idx=0, smoothing=0.1)
        >>> x = tf.random.uniform((10, 5000))  # model's output
        >>> target = tf.random.uniform((10, 1), maxval=5000, dtype=tf.int32)  # target labels
        >>> loss = loss_func(x, target)  # calculate loss
    """

    def __init__(self, vocab_size, padding_idx, smoothing=0.0):
        """
        Args:
            vocab_size (int): The size of the vocabulary, which also represents the number of classes.
            padding_idx (int): The index representing padding elements.
            smoothing (float): The smoothing factor to be applied. The values should be between 0 and 1. 
                            Default value is 0.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        # confidence is used for the position of the predicted token, while smoothing is applied to all not predicted tokens
        self.confidence = 1.0 - smoothing   # value for prediction
        self.smoothing = smoothing          # value for smoothing
        
    def call(self, prediction, target):
        """
        This function applies label smoothing to the target labels, computes the KL divergence loss 
        between the predicted and smoothed target distributions, then masks out the padding tokens 
        in the loss (since those should not contribute to the training signal). Finally, it averages 
        the loss over the non-padding tokens.

        Args:
            prediction (tf.Tensor):     The predicted token logits from the model in form of a one-hot-encoding tensor.
                                        Shape is [batch_size, sequence_length, vocab_size].
            target (tf.Tensor):         The target token IDs. Shape is [batch_size, sequence_length].

        Returns:
            loss (tf.Tensor):           The average loss (scalar) for the given batch.

        Note:
            The loss is averaged over non-padding tokens.
        """
        # create padding mask
        mask = self.padding_mask(target, self.padding_idx)

        # Apply label confidence
        true_dist = target * self.confidence

        # Apply label smoothing
        smoothing_value = self.smoothing / tf.cast(self.vocab_size - 2, tf.float32)
        true_dist = tf.where(tf.equal(true_dist, 0), smoothing_value, true_dist)

        # Calculate the loss
        kl_div_loss = self.kl_div_loss(prediction, true_dist)
        masked_loss = tf.cast(self.apply_mask(kl_div_loss, mask), prediction.dtype)
        loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(mask)

        return loss
    
    @staticmethod
    def padding_mask(tensor, padding_idx):
        """
        Create a binary mask where padding entries are 0 and others are 1.

        Args:
            tensor (tf.Tensor):     A tensor to be masked, of any shape.
            padding_idx (int):      The value that represents padding in the tensor.

        Returns:
            tf.Tensor:              A binary mask of the same shape as input tensor.
        """
        return tf.cast(tf.equal(tensor[:, :, padding_idx], 0), tf.float32)

    @staticmethod
    def apply_mask(tensor, mask):
        """
        Applies a mask to a tensor, zeroing out where the mask is on.

        Args:
            tensor (tf.Tensor):     A tensor to be masked, of any shape.
            mask (tf.Tensor):       A mask tensor, must be broadcastable to the shape of 'tensor'.

        Returns:
            tf.Tensor:              A tensor of the same shape as input tensor but with masked values zeroed out.
        """
        # mask stores padding bools in [batch_size, sequence_length] shape, 
        # we need to extend it to have shape [batch_size, sequence_length, vocab_size]
        expanded_mask = tf.broadcast_to(tf.expand_dims(mask, -1), tf.shape(tensor))

        return tensor * expanded_mask
    
    @staticmethod
    def kl_div_loss(input, target):
        """
        Calculates the Kullback-Leibler divergence between the input and target distributions.

        Notes: Inputs have to be logits, while target have to be probabilities.

        Args:
            input (tf.Tensor):      Input tensor, representing predicted probability distribution.
            target (tf.Tensor):     Target tensor, representing true probability distribution.

        Returns:
            tf.Tensor:              The KL divergence between the input and target distributions.
        """
        return target * (tf.math.log(target)-input)

# %%
class LossCompute(tf.keras.losses.Loss, VisualWrapper):
    """
    Custom loss computation class that computes loss on a batch of examples.
    This class inherits from tf.keras.losses.Loss, which allows it to be used seamlessly 
    within the Keras API.
    """
    def __init__(self, generator, loss_function, vocab_size, name='loss_compute'):
        """
        Initializes the LossCompute object.
        
        Args:
            generator (layers.Layer):       The generator layer.
            loss_function (layers.Layer):   The function class to compute the loss.
            vocab_size (int):               The size of the vocabulary.
            name (str, optional):           The name for the loss.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__(name=name)
        VisualWrapper.__init__(self, vis_on_count=[0])
        self.generator = generator
        self.loss_function = loss_function
        self.vocab_size = vocab_size

    def call(self, y_true, y_pred):
        """
        Computes the loss on a batch of examples.
        
        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.

        Returns:
            tf.Tensor: The total loss for the batch.
        """
        # generate predictions as one-hot encoded tensor
        y_pred = self.generator(y_pred)
        y_true_one_hot = tf.cast(tf.one_hot(y_true, depth=self.vocab_size), tf.float32)

        # Compute loss
        loss = self.loss_function(y_pred, y_true_one_hot)

        # Calculate mean loss per batch
        norm = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        sloss = loss / norm

        # Return total loss (for the whole batch)
        # TODO: Do we want mean loss or total loss?
        return loss

# %% [markdown]
# ### Learning rate and optimizer

# %%
class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule, VisualWrapper):
    """
    Custom learning rate scheduler for the Transformer model.

    This class inherits from tf.keras.optimizers.schedules.LearningRateSchedule, which allows it to be used seamlessly
    within the Keras API for dynamically adjusting the learning rate during training.

    It follows the learning rate schedule defined in the "Attention is All You Need" paper, which increases the 
    learning rate linearly for the first 'warmup_steps', and decreases it afterwards proportionally to the inverse 
    square root of the step number.
    """
    def __init__(self, d_model=512, warmup_steps=4000):
        """
        Initializes the TransformerSchedule object.
        
        Args:
            d_model (int, optional):        The dimensionality of the input.
            warmup_steps (int, optional):   The number of steps for the linear warmup phase.
        """
        log.debug(f'initialize {self.__class__.__name__}')
        super().__init__()
        VisualWrapper.__init__(self, vis_on_count=[0])

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32) # for calculations we need float tensors
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Computes the learning rate for a given step.
        
        Args:
            step (int): The current training step.

        Returns:
            tf.Tensor: The learning rate for the provided step.
        """
        step = tf.cast(step, dtype=tf.float32)  # convert for calculations
        arg_1 = tf.math.rsqrt(step)             # rsqrt is equivalent to 1/sqrt(x)
        arg_2 = step * (self.warmup_steps ** -1.5)
        
        # Minimum of two arguments provides the linear warmup phase for the first 'warmup_steps'
        # and the decrease proportional to the inverse square root of the step afterwards.
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg_1, arg_2)

# %% [markdown]
# ### Training metrics

# %%
def masked_accuracy(label, pred, pad_idx):
  """
  This function calculates the accuracy of the prediction while ignoring the specified padding index. 
  It assumes that labels have already been converted into indices (i.e., not one-hot encoded).

  Args:
      label (tf.Tensor):  The ground truth labels. These should be integer indices, 
                          not one-hot encoded, with a shape of (batch_size, seq_length).
      pred (tf.Tensor):   The predicted labels, given by the model. These should be 
                          the raw outputs of the model (i.e., logits) with a shape of 
                          (batch_size, seq_length, vocab_size).
      pad_idx (int):      The index representing the padding in the sequence. This will be excluded 
                          from the accuracy calculation.

  Returns:
      tf.Tensor: The accuracy of the model's predictions, excluding padding. 
                  It is a scalar tensor (0-dimensional).
  """

  pred = tf.argmax(pred, axis=2)      # calculate prediction tokens from logits
  label = tf.cast(label, pred.dtype)  # assure matching works

  match = label == pred   # mask
  mask = label != pad_idx # mask out padding
  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)

  return tf.reduce_sum(match)/tf.reduce_sum(mask)

def accuracy_with_pad_idx(pad_idx):
   """
   Returns an accuracy function where the pad_idx is already set.add

   Args:
      pad_idx (int): An id for the padding token such that it can be masked in accuracy calculation

    Returns:
      function: A accuracy function that compares label and prediction.
   """
   return lambda label, pred: masked_accuracy(label, pred, pad_idx)

# %% [markdown]
# ## Model creation

# %%
def make_model(src_vocab, 
               tgt_vocab, 
               N=6, 
               d_model=512, 
               d_ff=2048, 
               h=8, 
               dropout=0.1) -> tf.keras.Model:
    """
    Constructs a Transformer model from the given hyperparameters.

    Args:
        src_vocab (int):            The size of the source vocabulary.
        tgt_vocab (int):            The size of the target vocabulary.
        N (int, optional):          The number of layers in the Transformer's encoder and decoder stacks. Default is 6.
        d_model (int, optional):    The dimension of the Transformer's embedding space. Default is 512.
        d_ff (int, optional):       The dimension of the feed forward network model. Default is 2048.
        h (int, optional):          The number of attention heads. Default is 8.
        dropout (float, optional):  The dropout rate. Default is 0.1.

    Returns:
        model (tf.keras.Model): A Transformer model constructed from the provided hyperparameters.

    This function constructs an Encoder-Decoder model using the specified hyperparameters. 
    The Encoder and Decoder stacks each consist of N layers. 
    Each layer in the Encoder stack consists of a multi-headed self-attention mechanism, 
    followed by position-wise fully connected feed-forward network. 
    Each layer in the Decoder stack consists of a multi-headed self-attention mechanism, 
    a multi-headed source-attention mechanism over the Encoder's output, 
    and position-wise fully connected feed-forward network.
    """
    log.debug(f'execute')
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
    log.debug(f'model set up')
    return model

# %% [markdown]
# #### Model trainer

# %%
class ModelTrainer():
    """
    Documentation
    """
    def __init__(self, 
                 tokenizer, 
                 data_generator,
                 data_path = None,
                 dataset_size = None,
                 train_val_test_size = (0, 0, 0),
                 d_model = 512,
                 n_stacks = 6,
                 h_att = 8,
                 smoothing = 0.1,
                 max_padding = 512,
                 pad_idx = 0,
                 global_batch_size = 64,
                 warmup_steps = 4000,
                 base_lr = None,
                 n_epochs = 1,
                 initial_epoch = None,
                 verbosity = 2,
                 distributed_strategy = tf.distribute.MirroredStrategy(),
                 load_model = False,
                 save_model = True,
                 model_load_path = None
                 ):
        """
        Docstring
        """
        log.debug(f'initialize {self.__class__.__name__}')
        # class modules
        self.tokenizer = tokenizer

        # var for model compile
        self.vocab_size = tokenizer.get_vocab_size()
        self.d_model = d_model
        self.n_stacks = n_stacks
        self.h_att = h_att
        self.smoothing = smoothing
        self.accuracy = accuracy_with_pad_idx(pad_idx)

        # var for distributed training
        self.strategy = distributed_strategy
        self.n_devices = distributed_strategy.num_replicas_in_sync
        log.debug(f"number of processing devices = {self.n_devices}")

        # var for batching
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = (global_batch_size / self.n_devices)

        # var for data generation
        self.data_path = data_path
        if dataset_size is not None:
            self.train_val_test_size = tuple(map(round, (dataset_size * 8/10, dataset_size * 1/10, dataset_size * 1/10)))
        else:
            self.train_val_test_size = train_val_test_size
        self.max_padding = max_padding
        self.pad_idx = pad_idx
        self.data_generator = data_generator(tokenizer = tokenizer, 
                                             batch_size = self.global_batch_size,
                                             train_val_test_size = self.train_val_test_size,
                                             max_padding = max_padding,
                                             pad_id = pad_idx)
        
        self.train_data, self.val_data, self.test_data = self.load_dataset(self.data_generator,
                                                                            self.data_path,
                                                                            self.train_val_test_size)
        
        #if self.strategy is not None:
        #    multi_dev_data_gen = lambda x: self.data_generator.multi_device_generate_dataset(x, data_path,
        #                                           train_val_test_size)
        #    self.train_data = self.strategy.distribute_datasets_from_function(multi_dev_data_gen)
        #    self.val_data = self.train_data # TODO Change this !!!!!
        #else:    
            
        # var for model fit
        self.n_epochs = n_epochs
        self.initial_epoch = 0 or initial_epoch
        self.data_steps_total = round(self.train_val_test_size[0] / self.global_batch_size)
        self.validation_steps_total = round(self.train_val_test_size[1] / self.global_batch_size)
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.fit_verbosity = verbosity
        self.callbacks = []
        
        # var for load and save
        self.load_model = load_model
        self.save_model = save_model
        self.model_load_path = model_load_path

        # compile model and load model weights if applicable
        self.model = self.compile_model()
        if load_model:
            self.load_model_weights(self.model, self.d_model, self.model_load_path)
        
        # add checkpoints
        if save_model:
            self.add_save_checkpoints()

    def load_dataset(self, data_generator, data_path, train_val_test_size):
        """
        This function loads the dataset into the ModelTrainer.
        """
        log.debug(f'execute')
        if data_path is not None:
            return data_generator.generate_dataset(data_path,
                                                   train_val_test_size)
        else:
            return None, None, None
        
    def compile_model(self):
        with self.strategy.scope():
            # set_up_model
            model = make_model(self.vocab_size, 
                               self.vocab_size, 
                               d_model = self.d_model,
                               N = self.n_stacks,
                               h = self.h_att)
            log.debug(f'model set up')

            # compile model
            model.compile(
            loss = LossCompute(model.generator, 
                               LabelSmoothingLoss(self.vocab_size, 
                                                  self.pad_idx, 
                                                  self.smoothing), 
                               self.vocab_size), 
            optimizer = tf.keras.optimizers.Adam(TransformerSchedule(self.d_model, 
                                                                     self.warmup_steps), # type: ignore
                                                                     beta_1=0.9, 
                                                                     beta_2=0.98, 
                                                                     epsilon=1e-9), 
            metrics = [self.accuracy],
            )
            log.debug(f'model compiled')

            VisualWrapper.reset_counter()

        return model
    
    def run_model(self, training_data=None, validation_data=None, epochs=None):
        """
        Execute model training
        """
        training_data = training_data or self.train_data
        validation_data = validation_data or self.val_data
        epochs = epochs or self.n_epochs

        if training_data is None or validation_data is None or epochs is None:
            raise ValueError("Training data, validation data and epochs must be provided either as arguments or as instance attributes.")
        
        log.debug(f'execute model fit with {epochs} epochs')

        
        self.model.fit(training_data, 
                       epochs = epochs,
                       steps_per_epoch = 10, #self.steps_per_epoch,
                       validation_data = validation_data.take(10),
                       validation_steps = 10, #self.validation_steps,
                       callbacks = self.callbacks,
                       verbose = self.fit_verbosity)
        
        if self.save_model:
            self.save_model_weights()
        
        print(self.model.summary())
        
        VisualWrapper.reset_counter()
    
    def load_model_weights(self, model, d_model, model_folder):
        """
        Load the latest model weights if available.

        Args:
            model (tf.keras.Model):         The model to which the weights will be loaded.
            d_model (int):                  The dimension of the Transformer architecture.
            model_folder (str, optional):   The directory from which to load the weights. 
                                            Default is None.

        Returns:
            model (tf.keras.Model):         The model with the loaded weights.
            
        This function loads the weights from the latest trained model found in the provided model_folder 
        or from the latest model in the current directory if load_latest is True.
        """
        log.debug(f'execute')
        # TODO: Ensure architecture sizes match.
        if model_folder is not None:
            log.debug(f'model_folder={model_folder}')
            # Load weights from the specified model folder
            directories = [model_folder]
        else:
            directories = sorted(pathlib.Path('.').glob('model_N*_h*'), key=lambda x: x.stat().st_mtime, reverse=True)

        log.debug(f'load_dir={directories}')

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

        log.debug(f'model weights extracted')

        # Load weights if we found a previously trained model
        if latest_weights is not None:
            print(f'Loading weights from {latest_weights}')
            
            # Create a dummy input matching the input shape of the model
            # TODO: Ensure that the shape and type of the dummy_input match with the actual input that your model is going to receive.
            dummy_input = tf.random.uniform(shape=[1,d_model]), tf.random.uniform(shape=[1,d_model]), None, None
            # Call the model on the dummy input
            _ = model.generator(model(dummy_input))

            model.load_weights(latest_weights)
        log.debug(f'model loaded with weights')

    def add_save_checkpoints(self):
        log.debug(f'execute')

        current_time = time.strftime("%Y%m%d-%H%M%S")

        directory = f"model_N{self.n_stacks}_h{self.h_att}_d{self.d_model}_t{current_time}"
        ckp_name = "model_{epoch:03d}.h5"
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = dir_path / ckp_name
        
        epoch_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                              save_freq='epoch',
                                                              save_weights_only=True, 
                                                              verbose=1)
        
        self.callbacks.append(epoch_checkpoint)
    
    def save_model_weights(self):
        log.debug(f'execute')

        current_time = time.strftime("%Y%m%d-%H%M%S")

        directory = f"model_N{self.n_stacks}_h{self.h_att}_d{self.d_model}_t{current_time}"
        final_name = "final_model.h5"
        dir_path = pathlib.Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        save_path = dir_path / final_name

        self.model.save_weights(save_path, overwrite=True)



# %% [markdown]
# ### Run model trainer

# %%
model_trainer = ModelTrainer(StoryTokenizer(reserved_tokens, vocab_path),
                             DatasetGenerator,
                             bco_file_path,
                             dataset_size=3870000,
                             d_model=512,
                             n_stacks=6,
                             h_att=8,
                             max_padding=512,
                             global_batch_size=16,
                             warmup_steps=4000,
                             n_epochs=10,
                             initial_epoch=0,
                             verbosity='auto',
                             distributed_strategy=tf.distribute.MultiWorkerMirroredStrategy(),
                             load_model=False,
                             save_model=True,
                             model_load_path=None)

# %%
model_trainer.run_model()

# %% [markdown]
# # Inference

# %% [markdown]
# ## WordComplete model

# %%
class WordComplete(tf.Module, VisualWrapper):
  """
    This class defines a complete sequence generation model for a Transformer. 
    It uses a given tokenizer and Transformer model to generate sequences.
  """
  def __init__(self, 
               tokenizer, 
               transformer, 
               max_length=512, 
               dtype=tf.Tensor, 
               decode_result=True):
    """
    Args:
        tokenizer (Tokenizer):          Tokenizer object to convert raw text into tokens.
        transformer (tf.keras.Model):   A Transformer model used for sequence generation.
        max_length (int, optional):     The maximum length of sequences that can be generated.
                                        Default is 512.
        dtype (tf.Tensor, optional):    The datatype of the output tensor. Default is tf.Tensor.
        decode_result (bool, optional): If True, decode the output tensor into a string. 
                                        Default is True.
    """
    log.debug(f'initialize {self.__class__.__name__}')
    super().__init__()
    VisualWrapper.__init__(self, vis_on_count=None)
    self.tokenizer = tokenizer
    self.transformer = transformer
    self.max_length = max_length
    self.dtype = dtype
    self.decode_result = decode_result

  def __call__(self, input, decode=True, encoding='utf-8', training=None):
    """
    Performs the sequence generation.

    Args:
        input (str or tf.Tensor):   The input sequence.
        decode (bool, optional):    If True, the output sequence is decoded into a string. 
                                    Default is True.
        encoding (str, optional):   The encoding to use when decoding the output sequence. 
                                    Default is 'utf-8'.
        training (bool, optional):  Whether the model is currently training. Default is None.

    Returns:
        text (str or tf.Tensor):    The generated text. If decode_result is True, this is a string.
                                    Otherwise, it is a tensor.
        tokens (tf.Tensor):         The tensor of generated tokens.
    """
    VisualWrapper.should_visualize = True
    
    # TODO: Bug with empty strings as input
    # Convert input to tensor if it is not already
    # Create a dynamic tensor to store output
    # Make sure tensor_input is 2-D
    tensor_input = tf.convert_to_tensor(input)
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    if len(tensor_input.shape) == 0:
      tensor_input = tensor_input[tf.newaxis]

    # tokenize and encode input
    # Identify end token of the input
    tokenized_input = self.tokenizer.tokenize(tensor_input, training=training).to_tensor()
    context = self.transformer.encode(tokenized_input, None, training=training)
    end = tokenized_input[-1][-1]

    # Write the input tokens (excluding the last one) to the output array
    for i, value in enumerate(tokenized_input[0][:-1]):
      output_array = output_array.write(i, value)

    # Start the generation of sequence from the last position of the input to max_length
    for i in tf.range(output_array.size(), self.max_length):

      # Prepare input for decoder
      # Decode the input
      dec_input = output_array.concat()[tf.newaxis]
      decode = self.transformer.decode(context, None, dec_input, None, training=training)

      # Create logits predictions and select the last predicted token
      predictions = self.transformer.generator(decode, training=training)
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.
      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the decoder as its input again.
      output_array = output_array.write(i, predicted_id[0][0])

      # break the loop, if [End] token is predicted
      if predicted_id == end:
        break
    
    # Create a tensor for detokenization
    # Detokenize
    # Create tokens from detokenized output again
    output = output_array.concat()[tf.newaxis]
    text = self.tokenizer.detokenize(output)
    tokens = self.tokenizer.lookup(output)

    # If decode_result is True, decode the text tensor into a string
    if self.decode_result:
      text = text.numpy()[0].decode(encoding)

    # reset visualisation
    VisualWrapper.should_visualize = False
    VisualWrapper.reset_counter()

    return text, tokens

# %% [markdown]
# ## Text inference

# %%
inference_model = WordComplete(StoryTokenizer(reserved_tokens, vocab_path), model_trainer.model, max_length=32)

string = "What will be your"

text, tokens = inference_model(string)

print(text)