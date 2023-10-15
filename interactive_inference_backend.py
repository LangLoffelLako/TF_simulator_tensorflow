# %%
# logging and decorators
import logging as log
import psutil

# system tools
import pathlib

# general modules
import numpy as np

# tensorflow modules
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

# necessary for visualization and user input
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ipywidgets as widgets
from ipywidgets import interact_manual, interactive, interact, VBox, HTML
from IPython.display import display, clear_output

# %%
# logging settings
log.basicConfig(
    format='%(asctime)s %(levelname)-8s %(processName)s %(threadName)s %(funcName)-20s %(message)s',
        # log.INFO for normal run
    level=log.INFO,
        # log.DEBUG for diagnostics
    # level=log.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# paths
train_file_path = "datasets/bookscorpusopen/processed_512"
val_file_path = "datasets/corpus/processed_512"

vocab_path = 'datasets/vocab.txt'

# tokenizer
tokenizer_name = 'story_corpus_tokenizer'
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

# %%
def do_nothing(*args, **kwargs):
    """Placeholder for VisualWrapper"""
    pass

def clones(layer_class, N, **kwargs):
    """Produce N identical layers"""
    log.debug(f'execute with class {layer_class.__class__.__name__} and N={N}')
    return [layer_class(**kwargs) for _ in range(N)]

def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return subsequent_mask == 0

# %%
class VisualWrapper():
    """This is a mixin-Class for the tensorflow layers that enable visualization during non-training sessions."""
    instances = []          # save instances of VisualWrapper for reset_counter classmethod (see below)
    n_vis_layers_per_class = {
        'StoryTokenizer': 1,
        'EncoderDecoder': 1,
        'EncoderStack': 1,
        'MultiHeadedAttention': 1,
        'StoryTokenizer': 1,
        'PositionalEmbedding': 1,
        'Generator': 1,
        'ResidualSublayer': 1,
    }
    vis_data = []

    def __init__(self, vis_on_count=None):
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
        self.vis_on_count = vis_on_count if vis_on_count else [0]
        if type(self).__name__ in self.n_vis_layers_per_class:
            num_instances = sum(isinstance(obj, type(self)) for obj in VisualWrapper.instances)
            if num_instances < self.n_vis_layers_per_class[type(self).__name__]:
                log.debug(f'append {self} to VisualWrapper.instances')
                VisualWrapper.instances.append(self)

    def increase_count(self):
        """Increase counter"""
        log.debug(f'execute')
        self.counter += 1

    # TODO: Enter standard texts and labels.
    def save_data(self, 
                  text, 
                  x, 
                  mode_x, 
                  text_x,
                  y=None, 
                  z=None,
                  mode_y=None,  
                  mode_z=None,
                  text_y=None,
                  text_z=None, 
                  x_axis=None, 
                  y_axis=None,
                  id=None):
        """Saving data for visualization"""
        log.debug(f'execute')
        if self in self.instances:
            if self.counter in self.vis_on_count:
                self.increase_count()
                log.debug(f'append data to vis_data')
                self.vis_data.append({'x': x, 
                                    'y': y,
                                    'z': z,
                                    'mode_x': mode_x,
                                    'mode_y': mode_y,
                                    'mode_z': mode_z,
                                    'text': text,
                                    'text_x': text_x,
                                    'text_y': text_y,
                                    'text_z': text_z,
                                    'x_axis': x_axis,
                                    'y_axis': y_axis,
                                    'id':id})

    @classmethod         
    def vis_data_generator(cls, id=None):
        log.debug(f'initialize generator')
        for data in cls.vis_data:
            if id == None:
                log.debug(f'yield {data}')
                yield data
            elif data['id'] == id:
                log.debug(f'yield {data} with id {id}')
                yield data
    

    @classmethod
    def visualize_data(cls, id=None):
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
        
        vis_data_gen = cls.vis_data_generator(id=id)

        button = widgets.Button(description="Click to proceed")
        output = widgets.Output()

        def on_button_clicked(b):
            log.debug(f'execute')
            with output:
                # if all checks for visualization are passed execute visualisation

                try:
                    data = next(vis_data_gen)
                    text = data['text']
                    display_values = []
                    display_values.append((data['x'], data['mode_x'], data['text_x']))
                    display_values.append((data['y'], data['mode_y'], data['text_y']))
                    display_values.append((data['z'], data['mode_z'], data['text_z']))
                    
                    log.debug(f'visualise data: {data}')
        
                    # print explanatory text
                    cls.display_text(text)

                    for data in display_values:
                        # choose the correct visualization function
                        visualisation_func = cls.choose_func(data[1])
                        # print explanatory text
                        cls.display_text(data[2])
                        # apply visualization function to data_x
                        visualisation_func(data[0])
                except StopIteration:
                    log.debug(f'Vis data generator exhausted.')
                    b.disabled = True

        button.on_click(on_button_clicked)
        box = VBox([output, button])
        display(box)

    @classmethod    
    def choose_func(cls, mode):
        """
        This function returns an executable function for the chosen 'mode'.

        Args:
            mode (string): The string indicating the visualization mode to apply.

        Returns:
            function: An executable function taking one input argument. This argument should be the data to be visualized.
        """
        log.debug(f'execute')
        if mode == 'color_bar':
            return lambda x: cls.color_bar(x)
        elif mode == 'print':
            return lambda x: cls.display_text(x)
        elif mode == 'reduce_dim':
            return lambda x: cls.reduce_dim(x)
        elif mode == 'matrix':
            return lambda x: cls.matrix_repr(x)
        else:
            # return a placeholder function, if no valid 'mode' is given.
            return do_nothing

    @classmethod
    def display_text(cls, text):
        log.debug(f'execute')
        if isinstance(text, str):
            display(HTML('<p style="font-size:18px; color:blue;">' + text + '</p>'))

    @classmethod
    def color_bar(cls, tensor, xlabel=None, ylabel=None):
        """
        Use matplotlib to plot a colorbar that visualizes the values of a 1-D-tensor.

        Args:
            tensor (tf.tensor): The tensor to be visualized
        """ 
        log.debug(f'execute')
        # labels for the plot TODO: Generalize such that the labels are valid for all data types.
        x_label = xlabel or 'Tiefe'
        y_label = xlabel or 'Position'

        # Assuming data[0] is a numpy array.
        # If it's a ListWrapper or another list-like object, convert it to a numpy array.
        # TODO: Doesn't work. Check for error.
        data_array = np.array(tf.squeeze(tensor))

        # If the array is 1D, reshape it into a 2D array with one column
        if data_array.ndim != 2:
            log.error('Error: Expected a 1D tensor')
            return

        # Set the size of the plot (you can adjust the dimensions as needed)
        fig, ax = plt.subplots(figsize=(10, 2))

        # Use matshow to create a color-coded visualization
        cax = ax.matshow(data_array, cmap='jet', aspect='auto')

        # Add colorbar
        fig.colorbar(cax, label='Wertebereich')

        # Set labels
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Set x and y tick locations to the middle of the cells
        #ax.set_xticks(np.arange(data_array.shape[1]), minor=False)
        #ax.set_yticks(np.arange(data_array.shape[0]), minor=False)

        plt.show()

    @classmethod
    def matrix_repr(cls, matrix):
        log.debug(f'execute')
        matrix = np.array(tf.squeeze(matrix))

        # If the tensor is not 3D, print an error message and return
        if matrix.ndim != 3:
            log.error('Error: Expected a 3D tensor')
            return

        # Calculate the number of subplots
        n_plots = matrix.shape[0]

        # Define the subplot grid dimensions (trying to get a roughly square grid)
        n_rows = int(np.sqrt(n_plots))
        n_cols = n_plots // n_rows if n_plots % n_rows == 0 else n_plots // n_rows + 1

        # Create a figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Iterate over each matrix in the tensor
        for i in range(n_plots):
            # Create a color-coded visualization of the matrix
            im = axes[i].imshow(matrix[i, :, :], cmap='jet')
            axes[i].set_xlabel('Eingabe')
            axes[i].set_ylabel('Ausgabe')

            # Set the title of the plot to indicate which matrix is being visualized
            axes[i].set_title(f'Aufmerksamkeitskopf {i + 1}')

        # Add colorbar, associating with the last image created
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Wertebereich')

        plt.show()

    @classmethod
    def reduce_dim(cls, tensor):
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
    def reset_visualiser(cls):
        """Reset the counter for all instances of the class."""
        log.debug(f'execute')
        for instance in cls.instances:
            instance.counter = 0
        cls.vis_data = []

# %%
class InputOutputWidget():

    def __init__(self,
                 input_value, 
                 input_description,
                 button_description,
                 output_function):
        self.input = widgets.Text(value=input_value, 
                                  description=input_description,
                                  continuous_update=False,  # updates value only when you finish typing or hit "Enter"
                                  layout = widgets.Layout(width='auto', margin='0px 0px 10px 0px')
                                  )
        self.button = widgets.Button(description=button_description,
                                     layout = widgets.Layout(width='auto'))

        self.output = widgets.Output(layout = widgets.Layout(width='auto'))
        self.out_func = output_function

    def on_button_click(self, b):
        with self.output:
            self.output.clear_output()  # clear the previous output
            self.out_func()

    def display(self):
        self.button.on_click(self.on_button_click)
        display(self.input, self.button, self.out_func)

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
        VisualWrapper.__init__(self)

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
    
        input_emb_enc = self.enc_embed(enc_input)
        input_emb_dec = self.dec_embed(dec_input)
        self.save_data(text='Zuerst wird der Eingabetext in ein für das Modell verarbeitbares Embedding verwandelt.',
                       x=input_emb_enc,
                       mode_x='color_bar',
                       text_x='This is the embedding created by the encoder.',
                       y=input_emb_dec,
                       z=input_emb_dec-input_emb_enc,
                       mode_y='color_bar',
                       mode_z='color_bar',
                       text_y='This is the embedding created by the decoder.',
                       text_z='Here you can see how the two embeddings differ from each other.',
                       x_axis='X-Achse',
                       y_axis='Y-Achse')

        return self.decode(self.encode(enc_input, pad_mask, training), 
                           pad_mask,
                           dec_input, 
                           subseq_mask, training)

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
        VisualWrapper.__init__(self)

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

        sublayer_dropout = self.dropout(sublayer_out, training=True)

        self.save_data(
                       text="Visualize difference before/after dropout.",
                       x = sublayer_out,
                       mode_x='color_bar',
                       text_x='Das ist die Ausgabe einer der Transformerblöcke',
                       y = sublayer_dropout,
                       mode_y="color_bar",
                       text_y='Das sind die Veränderungen, die eine eingefügte Dropout-Layer einführt.',
                       z= sublayer_out-sublayer_dropout,
                       mode_z='color_bar',
                       text_z='Hier sieht man wie die Werte durch das Dropout verändert wurde.')
            
        # compute residual output by applying dropout to the sublayer output and adding to the input
        residual_out = input_tensor + self.dropout(sublayer_out, training=training)

        return residual_out

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
        VisualWrapper.__init__(self)
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

        self.save_data(text='Hier sieht man, welchen Unterschied die Anwendung der Layernorm am Ende des Encoders macht.', 
                       x=input_tensor, 
                       mode_x='color_bar', 
                       text_x='Dies ist die Ausgabe des Encoder bevor die Layernorm angewandt wird.', 
                       y=encoder_out, 
                       mode_y='color_bar', 
                       text_y= "Dies ist die Ausgabe des Encoders nach der Anwendung der Layernorm.",
                       z=input_tensor-encoder_out,
                       mode_z='color_bar',
                       text_z='Das ist der Unterschied.',
                       id='layer')

        return encoder_out

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
        VisualWrapper.__init__(self)
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
        VisualWrapper.__init__(self)
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
        VisualWrapper.__init__(self)
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

        self.save_data(text=f"Am Ende wird eine logarithmische Softmax-Layer genutzt, um die Ausgabe des Modells auf die Länge des Vokabulars zu erweitern.", 
                       x=tf.math.exp(result), 
                       mode_x='color_bar',
                       text_x='Hier sieht man die finalen Wahrscheinlichkeiten, die das Modell generiert.')
        return result

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

        att_out_unmasked, attn_unmasked = attention(query, key, value, mask=None, dropout=None, training=training)

        # Now we reverse the whole process and reshape the output into vectors of shape (nbatches, 1, d_model) again.
        att_out = tf.reshape(tf.transpose(att_out, perm=[0, 2, 1, 3]), (nbatches, -1, self.h * self.d_k))

        # visualization functions
        self.save_data(text=f"Man wendet wiederholt den Aufmerksamkeitsmechanismus auf das Embedding des Modells an", 
                       x = att_out, 
                       mode_x="color_bar",
                       text_x="Hier sehen wir die Ausgabe die bei der Anwendung auf das Embedding entsteht.",
                       y = self.attn,
                       mode_y="matrix",
                       text_y=f"Dafür werden mehrere Aufmerksamkeitsköpfe (in Matrixform) parallel auf das Embedding angewandt und danach linear in ein neuen Embeddingvektor überführt. Hier sehen sie die dabei genutzten Aufmerksamkeitsmatrizen.",
                       id='attention')

        # This finally mixes the results of the different heads together into one output vector
        linear_output = self.linear(att_out)

        return linear_output

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
        VisualWrapper.__init__(self)
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

        emb = self.embedding(input_token_vec) # is now a tensor of shape (batch_size, length, d_model)
        emb_scaled = emb * tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # This factor sets the relative scale of the embedding and positional_encoding
        emb_pos_enc = emb_scaled + self.pos_encoding[tf.newaxis, :length, :]
        y = self.dropout(emb_pos_enc)
    
        self.save_data(text=f"Das ist das Embedding wie es von {self.__class__.__name__} aus dem Input Byte-Pair Encoding erzeugt wird.", 
                       x=emb_scaled,
                       mode_x='color_bar',
                       text_x='Hier sieht man das Embedding ohne Positionsinformationen.',
                       y=y,
                       mode_y='color_bar', 
                       text_y='Das Positionale Encoding verändert das ursprüngliche Embedding, um Positionsinformationen hinzuzufügen.',
                       z=emb_pos_enc-emb_scaled,
                       mode_z='color_bar',
                       text_z='Hier sieht man welche Veränderung die Positionsinformationen dem ursprünglichen Embedding hinzufügen.'
                       )
        return y

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
        VisualWrapper.__init__(self)

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

        self.save_data(text=f"In einem ersten Schritt erstellt der Tokenizer ein Byte-Pair Encoding des Satzes",
                       x=out,
                       mode_x='print',
                       text_x='Das ist die erzeugte Tokenliste.')

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

        self.save_data(text=f"Zuletzt werden die Daten durch einen Tokenizer in Textform überführt.",
                       x=self.lookup(tokenized),
                       mode_x='print', 
                       text_x="Dies ist also die Vorhersage, die das Modell für die unterschiedlichen Positionen liefert."
                       )

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

# %%
class ModelLoader():
    """
    Documentation
    """
    def __init__(self, 
                 tokenizer,
                 d_model = 512,
                 n_stacks = 6,
                 h_att = 8,
                 load_model = False,
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
        
        # var for load and save
        self.load_model = load_model
        self.model_load_path = model_load_path

        # compile model and load model weights if applicable
        self.model = self.set_up_model()
        if load_model:
            self.load_model_weights(self.model, self.d_model, self.model_load_path)
        
    def set_up_model(self):
        # set_up_model
        model = make_model(self.vocab_size, 
                            self.vocab_size, 
                            d_model = self.d_model,
                            N = self.n_stacks,
                            h = self.h_att)
        log.debug(f'model set up')

        VisualWrapper.reset_visualiser()

        return model

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
            directories = [pathlib.Path(model_folder)]
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
                latest_weights = h5_files[-1]

        log.debug(f'model weights extracted')

        # Load weights if we found a previously trained model
        if latest_weights is not None:
            log.debug(f'Loading weights from {latest_weights}')
            
            # Create a dummy input matching the input shape of the model
            # TODO: Ensure that the shape and type of the dummy_input match with the actual input that your model is going to receive.
            dummy_input = tf.random.uniform(shape=[1,d_model]), tf.random.uniform(shape=[1,d_model]), None, None
            # Call the model on the dummy input
            _ = model.generator(model(dummy_input))

            model.load_weights(latest_weights)
        log.debug(f'model loaded with weights')



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
               pad_id=0,
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
    VisualWrapper.__init__(self)
    self.tokenizer = tokenizer
    self.transformer = transformer
    self.max_length = max_length
    self.pad_id = pad_id
    self.dtype = dtype
    self.decode_result = decode_result
  
  def __call__(self, input, decode=True, encoding='utf-8', interactive=False):
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
    # during model set-up visualise data is created
    VisualWrapper.reset_visualiser()

    # initialize loading widget
    if interactive:
      load_bar = widgets.FloatProgress(value=0,
                                       min=0,
                                       max=self.max_length,
                                       description='Lädt',
                                       bar_style='info',
                                       style={'bar_color': 'green'},
                                       orientation='horizontal')
      display(load_bar)

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
    tokenized_input = self.tokenizer.tokenize(tensor_input).to_tensor()
    input_without_eos = tokenized_input[:, :-1]
    context = self.transformer.encode(input_without_eos, None)
    end = tokenized_input[-1][-1]

    # Write the input tokens (excluding the last one) to the output array
    for i, value in enumerate(tokenized_input[0][:-1]):
      output_array = output_array.write(i, value)

    # Start the generation of sequence from the last position of the input to max_length
    for i in tf.range(output_array.size(), self.max_length):
    
      if interactive:
        load_bar.value=i

      # Prepare input for decoder
      # Decode the input
      dec_input = output_array.concat()[tf.newaxis]

      decode = self.transformer.decode(context, None, dec_input, None)

      # Create logits predictions and select the last predicted token
      predictions = self.transformer.generator(decode)
      predictions_last = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.
      predicted_id = tf.argmax(predictions_last, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the decoder as its input again.
      output_array = output_array.write(i, predicted_id[0][0])

      # break the loop, if [End] token is predicted
      if predicted_id == end:
        break
    
    if interactive:
      load_bar.value = load_bar.max
    # Create a tensor for detokenization
    # Detokenize
    # Create tokens from detokenized output again
    output = output_array.concat()[tf.newaxis]
    text = self.tokenizer.detokenize(output)
    tokens = self.tokenizer.lookup(output)

    # If decode_result is True, decode the text tensor into a string
    if self.decode_result:
      text = text.numpy()[0].decode(encoding)
      print(text)
  
  def print_results(self, visualisation=False):
    if visualisation:
      VisualWrapper.visualize_data()
