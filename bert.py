import math
import numpy as np
import ujson as json
import six
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Input, Embedding, Softmax, Add
from keras.initializers import TruncatedNormal
from keras import backend as K
from keras.engine.topology import Layer

if K.backend() == "cntk":
    import cntk as C

def batch_dot(a, b):
    if K.backend() == "cntk":
        a_shape = K.shape(a)
        a = K.reshape(a, [-1] + list(a_shape[-2:]))

        b_shape = K.shape(b)
        b = K.reshape(b, [-1] + list(b_shape[-2:]))

        res = C.times(a, b)
        return K.reshape(res, [-1] + list(a_shape[1:-1]) + [b_shape[-1]])
    else:
        return K.batch_dot(a,b)

def swapaxes(t, axis1, axis2):
    if K.backend() == "cntk":
        return C.swapaxes(t, axis1=(axis1-1), axis2=(axis2-1)) # 0, 3, 2, 1, 4
    else:
        swap = np.arange(K.ndim(t))
        swap[axis1] = axis2
        swap[axis2] = axis1
        return K.permute_dimensions(t, swap)

def erf(x):
    positives = K.cast(K.greater_equal(x, 0.), K.dtype(x))
    sign =  positives * 1 + (1. - positives ) * -1
    x = K.abs(x)
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t* K.exp(-x*x)
    return sign*y


class GELU(Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, x):
        cdf = 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
        return x * cdf

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer='one', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(input_shape[-1],), initializer='zero', trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, mask=None, training=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) / K.sqrt(variance + K.epsilon())
        return norm_x * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

class Reduce(Layer):
    def __init__(self, mode="first", **kwargs):
        self.mode = mode
        super(Reduce, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Reduce, self).build(input_shape)

    def call(self, x):
        x, mask = x
        mask = K.expand_dims(mask, axis=-1)

        if self.mode == "first":
            return K.squeeze(x[:, 0:1, :], axis=1)
        elif self.mode == "max+":
            return K.max(K.maximum(x*mask, 0.), axis=1)
        elif self.mode == "mean":
            return K.sum(x*mask, axis=1, keepdims=False) / K.sum(mask, axis=1, keepdims=False)
        elif self.mode == "max1": # without first value
            return K.max(K.maximum(x[:, 1:, :]*mask[:, 1:, :], 0.), axis=1)
        elif self.mode == "max": # with negatives
            return K.max(x*mask - 1000000 * (1 - mask), axis=1)
        elif self.mode == "mean1": # remove first value
            return K.sum(x[:, 1:, :]*mask[:, 1:, :], axis=1, keepdims=False) / K.sum(mask[:, 1:, :], axis=1, keepdims=False) # remove first value TODO

        else:
            raise ValueError("Reduce mode %s not known." % self.mode)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])


class AttentionMask(Layer):
    def __init__(self, **kwargs):
        super(AttentionMask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionMask, self).build(input_shape)

    def call(self, x):
        from_tensor, to_mask = x
        to_mask = K.cast( K.reshape(to_mask, [-1, 1, K.shape(to_mask)[1] ]),  K.floatx())
        mask = K.tile(to_mask, (1, K.shape(from_tensor)[1], 1))
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = K.int_shape(input_tensor)
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                        (input_width, hidden_size))

    prev_output = Reshape((-1, input_width))(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        layer_input = prev_output

        # `query_layer` = [B*F, N*H]
        query_layer = Dense(num_attention_heads * attention_head_size,
                            activation=None,
                            name="encoder/layer_%d/attention/self/query" % (layer_idx),
                            kernel_initializer=TruncatedNormal(stddev=initializer_range))(layer_input)
        # return query_layer
        # `key_layer` = [B*T, N*H]
        key_layer = Dense(num_attention_heads * attention_head_size,
                            activation=None,
                            name="encoder/layer_%d/attention/self/key" % (layer_idx),
                            kernel_initializer=TruncatedNormal(stddev=initializer_range))(layer_input)

        # `value_layer` = [B*T, N*H]
        value_layer = Dense(num_attention_heads * attention_head_size,
                            activation=None,
                            name="encoder/layer_%d/attention/self/value" % (layer_idx),
                            kernel_initializer=TruncatedNormal(stddev=initializer_range))(layer_input)

        attention_output = AttentionLayer(
            num_attention_heads=num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=False,
            seq_length=seq_length)([query_layer, key_layer, value_layer, attention_mask])

        attention_output = Dense(hidden_size,
                                    name="encoder/layer_%d/attention/output/dense" % layer_idx,
                                    kernel_initializer=TruncatedNormal(stddev=initializer_range))(attention_output)
        attention_output = Dropout(hidden_dropout_prob)(attention_output)
        attention_output = Add()([attention_output, layer_input])
        attention_output = LayerNormalization(name = ("encoder/layer_%d/attention/output" % layer_idx) + "/LayerNorm")(attention_output)

        # The activation is only applied to the "intermediate" hidden layer.
        intermediate_output = Dense(
            intermediate_size,
            name = "encoder/layer_%d/intermediate/dense" % layer_idx,
            kernel_initializer=TruncatedNormal(initializer_range))(attention_output)
        intermediate_output = GELU()(intermediate_output)

        # Down-project back to `hidden_size` then add the residual.
        layer_output = Dense(
            hidden_size,
            name="encoder/layer_%d/output/dense" % layer_idx,
            kernel_initializer=TruncatedNormal(initializer_range))(intermediate_output)

        layer_output = Dropout(hidden_dropout_prob)(layer_output)
        layer_output = Add()([layer_output, attention_output])
        layer_output = LayerNormalization(name=("encoder/layer_%d/output" % layer_idx) + "/LayerNorm")(layer_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = layer_output
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = layer_output
        return final_output


class AttentionLayer(Layer):
    def __init__(self, num_attention_heads=1,
                        size_per_head=512,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        seq_length=None,
                        **kwargs):
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.seq_length = seq_length
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.do_return_2d_tensor = do_return_2d_tensor
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, training=None):
        query_layer, key_layer, value_layer, attention_mask = x

        def transpose_for_scores(input_tensor, num_attention_heads, seq_length, width):
            output_tensor = K.reshape( input_tensor, [-1, seq_length, num_attention_heads, width])

            return swapaxes(output_tensor, 1, 2)

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer, self.num_attention_heads, self.seq_length, self.size_per_head)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, self.num_attention_heads, self.seq_length, self.size_per_head)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        key_layer = swapaxes(key_layer, 2, 3)
        attention_scores = batch_dot(query_layer, key_layer)
        attention_scores /= math.sqrt(float(self.size_per_head))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = K.expand_dims(attention_mask, axis=1)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - K.cast(attention_mask, K.floatx() )) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        attention_shape = K.shape(attention_scores)
        attention_scores = K.reshape(attention_scores, (-1, attention_shape[-1]))
        attention_probs = K.softmax(attention_scores)
        # `attention_probs` = [B, N, F, T]
        attention_probs = K.reshape(attention_probs, [-1, self.num_attention_heads, self.seq_length, self.seq_length])

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = K.in_train_phase(K.dropout(attention_probs, self.attention_probs_dropout_prob), attention_probs, training=training)


        # `value_layer` = [B, T, N, H]
        value_layer = K.reshape(value_layer, [-1, self.seq_length, self.num_attention_heads, self.size_per_head])

        # `value_layer` = [B, N, T, H]
        value_layer = swapaxes( value_layer, 1, 2)

        # `context_layer` = [B, N, F, H]
        context_layer = batch_dot(attention_probs, value_layer)

        # `context_layer` = [B, F, N, H]
        context_layer = swapaxes(context_layer, 1, 2)


        if self.do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = K.reshape(context_layer, [-1, self.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = K.reshape(context_layer, [-1, self.seq_length, self.num_attention_heads * self.size_per_head])

        return context_layer

    def compute_output_shape(self, input_shape):
        return (-1, self.seq_length, self.num_attention_heads * self.size_per_head)



class Network:

    def __init__(self, unknownargs, **kwargs):
        self.hidden_size=768,
        self.num_hidden_layers=12,
        self.num_attention_heads=12,
        self.intermediate_size=3072,
        self.hidden_dropout_prob=0.1,
        self.attention_probs_dropout_prob=0.1,
        self.max_position_embeddings=512,
        self.type_vocab_size=16,
        self.initializer_range=0.02

        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--bert_config_file', type=str,
            default="/sharedfiles/multi_cased_L-12_H-768_A-12/bert_config.json",
            help=   "The config json file corresponding to the pre-trained BERT model. "
                    "This specifies the model architecture.")
        parser.add_argument("--init_checkpoint", type=str, default="",
            help =  "Initial checkpoint (usually from a pre-trained BERT model).")
        parser.add_argument("--reduction", type=str, default="first",
            help =  "Reduction method.")


        newargs, _ = parser.parse_known_args(unknownargs)
        print("Model parameters:")
        for arg in vars(newargs):
            print(" "*3, arg +":", getattr(newargs, arg))
        self.__dict__.update(vars(newargs))

        print("Loading parameters from Bert config file:")
        with open(self.bert_config_file, "r") as reader:
            json_object = json.loads(reader.read())
            for (key, value) in six.iteritems(json_object):
                print(" "*3, key +":", value)
                self.__dict__[key] = value

        if self.hidden_act != "gelu":
            raise ValueError("Only GELU activation implemented.")

        self.strides = None
        self.offsets = None
        self.fields = None


    def build(self, seq_length = 128, **kwargs):

        # Input Embedding Layer
        input_ids = Input((seq_length,)) # [bs, len,]
        input_mask = Input((seq_length,)) # [bs, len]
        token_type_ids = Input((seq_length,)) # [bs, len]
        input_index = Input((seq_length,)) # [bs, len]
        inputs = [input_ids , input_mask, token_type_ids, input_index]

        # Word embeddings
        WordEmbedding = Embedding(self.vocab_size, self.hidden_size,
                                    trainable=False,
                                    embeddings_initializer=TruncatedNormal(stddev=self.initializer_range),
                                    name="embeddings/word_embeddings")

        embedding_output = WordEmbedding(input_ids)

        if seq_length > self.max_position_embeddings:
            raise ValueError("Sequence length has to be smaller than position embeddings length")

        # Token type embeddings
        TokenTypeEmbedding = Embedding(self.type_vocab_size, self.hidden_size,
                                    trainable=False,
                                    embeddings_initializer=TruncatedNormal(stddev=self.initializer_range),
                                    name="embeddings/token_type_embeddings")
        token_type_embeddings = TokenTypeEmbedding(token_type_ids)

        # Position embeddings
        FullPositionEmbedding = Embedding(seq_length, self.hidden_size,
                                    trainable=False,
                                    embeddings_initializer=TruncatedNormal(stddev=self.initializer_range),
                                    name="embeddings/position_embeddings")
        position_embeddings = FullPositionEmbedding(input_index)

        embedding_output = Add()([embedding_output, token_type_embeddings, position_embeddings])

        embedding_output = LayerNormalization(name = "embeddings/LayerNorm")(embedding_output)
        self.embedding_output = Dropout(self.hidden_dropout_prob)(embedding_output)

        attention_mask = AttentionMask()([input_ids, input_mask])

        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            do_return_all_layers=True)

        self.sequence_output = self.all_encoder_layers[-1]

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        self.token_tensor = Reduce(mode=self.reduction)([self.sequence_output, input_mask])
        self.pooled_output = Dense(self.hidden_size,
                                    activation="tanh",
                                    name="pooler/dense",
                                    kernel_initializer=TruncatedNormal(self.initializer_range))(self.token_tensor)

        self.model = Model(inputs=inputs, outputs=[self.pooled_output, self.token_tensor])

        if self.init_checkpoint != "":
            self.load_from_checkpoint(self.init_checkpoint)

        self.model.strides = None
        self.model.offsets = None
        self.model.fields = None
        return self.model


    def load_from_checkpoint(self, checkpoint):
        if K.backend() != "tensorflow":
            raise ValueError("Checkpoint available under Tensorflow backend only.")

        print("Loading from TF checkpoint")
        from tensorflow.python import pywrap_tensorflow
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        weights = {}
        for key in sorted(var_to_shape_map):
            tensor = reader.get_tensor(key)
            weights[key] = tensor

        scope = "bert"
        for layer in self.model.layers:
            skip = True

            if "/LayerNorm" in layer.name:
                W = [weights[scope + "/" + layer.name + "/gamma"  ], weights[scope + "/" + layer.name + "/beta"  ]]
                print("Setting weights for", layer.name, [ o.shape for o in layer.get_weights()], [w.shape for w in W])
                layer.set_weights(W)
                skip = False

            elif "embeddings" in layer.name:
                w = weights[scope + "/" + layer.name ]
                print("Setting weights for", layer.name, [ o.shape for o in layer.get_weights()], w.shape)
                dim = layer.get_weights()[0].shape[0]
                if w.shape[0] != dim:
                    print("Resizing embedding first dimension")
                    w = w[:dim]
                layer.set_weights([w])
                skip = False

            elif scope + "/" + layer.name + "/kernel" in weights:
                W = [weights[scope + "/" + layer.name + "/kernel" ], weights[scope + "/" + layer.name + "/bias" ]]
                print("Setting weights for", layer.name, [ o.shape for o in layer.get_weights()], [w.shape for w in W])
                layer.set_weights(W)
                skip = False

            if skip and len(layer.get_weights()):
                print("Skipping", layer.name)
