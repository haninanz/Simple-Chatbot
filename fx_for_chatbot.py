import re
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.seq2seq import prepare_attention, attention_decoder_fn_train, attention_decoder_fn_inference, dynamic_rnn_decoder
from tensorflow.contrib.layers import fully_connected, embed_sequence

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"could not", "could not", text)
    text = re.sub(r"\'bout", "about", text)
    text = re.sub(r"[-()\"#/@&;:<>{}+=~|.?,*^%!]", "", text)
    return text

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    target = tf.placeholder(tf.int32, [None, None], name = 'input')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, target, learning_rate, keep_prob

def preprocess_targets(targets, map_dictionary, batch_size):
    left_side = tf.fill(dims = [batch_size, 1],
                        value = map_dictionary['<SOS>'])
    right_side = tf.strided_slice(input_ = targets, begin = [0, 0],
                                  end = [batch_size, -1], strides = [1, 1])
    return tf.concat([left_side, right_side], axis = 1)

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, seq_length):
    lstm = BasicLSTMCell(num_units = rnn_size)
    lstm_dropout = DropoutWrapper(cell = lstm, input_keep_prob = keep_prob)
    encoder_cell = MultiRNNCell(cells = [lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = seq_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                        seq_length, decoding_scope, output_function,
                        keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_fx, attention_construct_fx = prepare_attention(attention_states,
                                                                                                     attention_option = 'bahdanau',
                                                                                                     num_units = decoder_cell.output_size)
    training_decoder_fx = attention_decoder_fn_train(encoder_state = encoder_state[0],
                                                     attention_keys = attention_keys,
                                                     attention_values = attention_values,
                                                     attention_score_fn = attention_score_fx,
                                                     attention_construct_fn = attention_construct_fx,
                                                     name = 'attn_dec_train')
    decoder_output, _, _ = dynamic_rnn_decoder(cell = decoder_cell,
                                               decoder_fn = training_decoder_fx,
                                               inputs = decoder_embedded_input,
                                               sequence_length = seq_length,
                                               scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(x = decoder_output,
                                           keep_prob = keep_prob)
    return output_function(decoder_output_dropout)

def decode_validation_set(encoder_state, decoder_cell, decoder_embeddings_matrix,
                          sos_id, eos_id, max_length, num_words, decoding_scope,
                          output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_fx, attention_construct_fx = prepare_attention(attention_states,
                                                                                                     attention_option = 'bahdanau',
                                                                                                     num_units = decoder_cell.output_size)
    validate_decoder_fx = attention_decoder_fn_inference(output_fn = output_function,
                                                         encoder_state = encoder_state[0],
                                                         attention_keys = attention_keys,
                                                         attention_values = attention_values,
                                                         attention_score_fn = attention_score_fx,
                                                         attention_construct_fn = attention_construct_fx,
                                                         embeddings = decoder_embeddings_matrix,
                                                         start_of_sequence_id = sos_id,
                                                         end_of_sequence_id = eos_id,
                                                         maximum_length = max_length,
                                                         num_decoder_symbols = num_words,
                                                         name = 'attn_dec_inf')
    predictions, _, _ = dynamic_rnn_decoder(cell = decoder_cell,
                                            decoder_fn = validate_decoder_fx,
                                            scope = decoding_scope)
    return predictions

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix,
                encoder_state, num_words, seq_length, rnn_size, num_layers,
                word2int_dict, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = BasicLSTMCell(num_units = rnn_size)
        lstm_dropout = DropoutWrapper(cell = lstm, input_keep_prob = keep_prob)
        decoder_cell = MultiRNNCell(cells = [lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: fully_connected(x, num_words, None,
                                                    scope = decoding_scope,
                                                    weights_initializer = weights,
                                                    biases_initializer = biases)
        train_pred = decode_training_set(encoder_state, decoder_cell,
                                         decoder_embedded_input, seq_length,
                                         decoding_scope, output_function,
                                         keep_prob, batch_size)
        decoding_scope.reuse_variables()
        test_pred = decode_validation_set(encoder_state, decoder_cell,
                                          decoder_embeddings_matrix, word2int_dict['<SOS>'],
                                          word2int_dict['<EOS>'], seq_length - 1,
                                          num_words, decoding_scope, output_function,
                                          keep_prob, batch_size)
    return train_pred, test_pred

def seq2seq_model(inputs, targets, keep_prob, batch_size, seq_length,
                  answers_num_words, questions_num_words, encoder_embedding_size,
                  decoder_embedding_size, rnn_size, num_layers,
                  questionswords2int_dict):
    encoder_embedded_input = embed_sequence(ids = inputs, vocab_size = answers_num_words + 1,
                                            embed_dim = encoder_embedding_size,
                                            initializer = tf.random_uniform_initializer(minval = 0, maxval = 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers,
                                keep_prob, seq_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int_dict,
                                              batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform(shape = [questions_num_words + 1, decoder_embedding_size],
                                                              minval = 0,
                                                              maxval = 1))
    decoder_embedded_input = tf.nn.embedding_lookup(params = decoder_embeddings_matrix,
                                                    ids = preprocessed_targets)
    train_pred, test_pred = decoder_rnn(decoder_embedded_input,
                                        decoder_embeddings_matrix,
                                        encoder_state, num_words, seq_length,
                                        rnn_size, num_layers,
                                        questionswords2int_dict, keep_prob,
                                        batch_size)
    return train_pred, test_pred
    